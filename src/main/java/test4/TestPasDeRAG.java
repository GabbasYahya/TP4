package test4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 4 - RAG ou pas RAG
 * QueryRouter personnalisé qui décide d'utiliser le RAG uniquement 
 * pour les questions pertinentes sur l'IA
 */
public class TestPasDeRAG {

    
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    /**
     * Ingère un document et retourne l'EmbeddingStore
     */
    private static EmbeddingStore<TextSegment> ingestDocument(
            Path documentPath,
            EmbeddingModel embeddingModel,
            DocumentParser parser,
            DocumentSplitter splitter) {

        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);
        List<TextSegment> segments = splitter.split(document);
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);

        return embeddingStore;
    }

    /**
     * Crée un ContentRetriever
     */
    private static ContentRetriever createContentRetriever(
            EmbeddingStore<TextSegment> embeddingStore,
            EmbeddingModel embeddingModel) {

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    public static void main(String[] args) {
        configureLogger();

        String geminiApiKey = System.getenv("GEMINI_KEY");
        if (geminiApiKey == null || geminiApiKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        // Configuration du modèle Gemini
        String modelName = System.getenv("GEMINI_MODEL");
        if (modelName == null || modelName.isBlank()) {
            modelName = "gemini-2.5-flash";
        }

        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName(modelName)
                .temperature(0.7)
                .logRequestsAndResponses(true)
                .build();

        // PHASE 1 : Ingestion du document sur l'IA
        Path documentIA = Paths.get("src/main/resources/support_rag.pdf");
        DocumentParser parser = new ApacheTikaDocumentParser();
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore = ingestDocument(
                documentIA, embeddingModel, parser, splitter);
        ContentRetriever contentRetriever = createContentRetriever(embeddingStore, embeddingModel);

        // PHASE 2 : Configuration du QueryRouter personnalisé

        // Template de prompt pour décider si la question porte sur l'IA
        final PromptTemplate promptTemplate = PromptTemplate.from(
                "Est-ce que la requête '{{question}}' porte sur l'intelligence artificielle, " +
                "le RAG, les embeddings, les modèles de langage ou des sujets techniques liés à l'IA ? " +
                "Réponds seulement par 'oui', 'non' ou 'peut-être'."
        );

        // Variable finale pour utilisation dans la classe anonyme
        final ContentRetriever finalContentRetriever = contentRetriever;

        // QueryRouter personnalisé avec classe anonyme
        // Ce router décide d'utiliser le RAG ou non selon le sujet de la question
        QueryRouter queryRouter = new QueryRouter() {
            @Override
            public Collection<ContentRetriever> route(Query query) {
                // Création du prompt avec la question de l'utilisateur
                Map<String, Object> variables = new HashMap<>();
                variables.put("question", query.text());
                Prompt prompt = promptTemplate.apply(variables);

                // Demande au LLM si la question porte sur l'IA
                String answer = chatModel.generate(prompt.text()).trim().toLowerCase();

                // Stratégie : "oui" ou "peut-être" active le RAG
                if (answer.contains("oui") || answer.contains("peut-être") || answer.contains("peut-etre")) {
                    // RAG activé : retourne le ContentRetriever
                    return Collections.singletonList(finalContentRetriever);
                }

                // RAG désactivé : retourne une liste vide
                return Collections.emptyList();
            }
        };

        // Configuration du RetrievalAugmentor avec le QueryRouter personnalisé
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Configuration de l'assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Boucle de questions-réponses
        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("Votre question : ");
            String question = scanner.nextLine().trim();

            if (question.equalsIgnoreCase("quitter") || question.equalsIgnoreCase("exit")) {
                System.out.println("\nAu revoir !");
                break;
            }

            if (question.isEmpty()) {
                continue;
            }

            try {
                String reponse = assistant.chat(question);
                System.out.println("\nRéponse : " + reponse + "\n");
            } catch (Exception e) {
                System.err.println("Erreur : " + e.getMessage());
                e.printStackTrace();
            }
        }
        scanner.close();
    }
}
