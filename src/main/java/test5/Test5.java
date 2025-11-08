package test5;

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
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 5 - RAG avec récupération sur le Web
 * Combine la recherche dans les documents locaux (PDF) et sur le Web (Tavily)
 */
public class Test5 {

    /**
     * Configure le logger pour voir les détails des recherches
     */
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

    public static void main(String[] args) {
        // Configuration du logging
        configureLogger();

        // Vérification des clés API
        String geminiApiKey = System.getenv("GEMINI_KEY");
        if (geminiApiKey == null || geminiApiKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        String tavilyApiKey = System.getenv("TAVILY_API_KEY");
        if (tavilyApiKey == null || tavilyApiKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement TAVILY_API_KEY n'est pas définie.");
        
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

        // PHASE 1 : Ingestion du document local (PDF)
        Path documentPath = Paths.get("src/main/resources/support_rag.pdf");
        DocumentParser parser = new ApacheTikaDocumentParser();
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore = ingestDocument(
                documentPath, embeddingModel, parser, splitter);

        // ContentRetriever pour le document local
        ContentRetriever documentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // PHASE 2 : Configuration de la recherche Web avec Tavily
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyApiKey)
                .build();

        // ContentRetriever pour la recherche Web
        ContentRetriever webSearchRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .build();

        // PHASE 3 : Configuration du QueryRouter qui utilise les deux sources
        // DefaultQueryRouter va utiliser les deux ContentRetrievers en parallèle
        QueryRouter queryRouter = new DefaultQueryRouter(documentRetriever, webSearchRetriever);

        // Configuration du RetrievalAugmentor avec le QueryRouter
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Configuration de l'assistant avec le RAG hybride (local + Web)
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        // Boucle de questions-réponses
        Scanner scanner = new Scanner(System.in);
        System.out.println("Assistant RAG hybride (Documents locaux + Web)");
        System.out.println("Tapez 'quitter' ou 'exit' pour terminer\n");

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

