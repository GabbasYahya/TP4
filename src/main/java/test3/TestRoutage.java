package test3;

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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 3 - Routage intelligent avec LLM
 * Utilise un QueryRouter basé sur un LLM pour sélectionner automatiquement
 * la source de documents la plus pertinente selon la question posée.
 */
public class TestRoutage {

    /**
     * Configure le logger pour afficher les détails du routage
     */
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    /**
     * Ingère un document : chargement, découpage et création des embeddings
     * 
     * @param documentPath Chemin vers le document à ingérer
     * @param embeddingModel Modèle d'embedding à utiliser
     * @param parser Parser pour le document
     * @param splitter Splitter pour découper le document
     * @return EmbeddingStore contenant les embeddings du document
     */
    private static EmbeddingStore<TextSegment> ingestDocument(
            Path documentPath,
            EmbeddingModel embeddingModel,
            DocumentParser parser,
            DocumentSplitter splitter) {

        System.out.println("Ingestion du document : " + documentPath.getFileName());
        
        Document document = FileSystemDocumentLoader.loadDocument(documentPath, parser);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("  - " + segments.size() + " segments créés");
        
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        System.out.println("  - " + embeddings.size() + " embeddings générés");
        
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        
        return embeddingStore;
    }

    /**
     * Crée un ContentRetriever à partir d'un EmbeddingStore
     * 
     * @param embeddingStore Store contenant les embeddings
     * @param embeddingModel Modèle d'embedding
     * @return ContentRetriever configuré
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
        // Configuration du logging pour voir le routage en action
        configureLogger();
        System.out.println("=== Test 3 : Routage Intelligent avec LLM ===\n");

        // Vérification de la clé API
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

        System.out.println("=== PHASE 1 : Ingestion des documents ===\n");

        // Définition des chemins des deux documents
        Path documentRAG = Paths.get("src/main/resources/support_rag.pdf");
        Path documentCyber = Paths.get("src/main/resources/Introduction to Cybersecurity v3.0 - Module1 - Introduction à la cybersécurité.pdf");

        // Composants partagés
        DocumentParser parser = new ApacheTikaDocumentParser();
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Ingestion des deux documents dans des stores séparés
        EmbeddingStore<TextSegment> embeddingStoreRAG = ingestDocument(
                documentRAG, embeddingModel, parser, splitter);
        
        EmbeddingStore<TextSegment> embeddingStoreCyber = ingestDocument(
                documentCyber, embeddingModel, parser, splitter);

        System.out.println("\n=== PHASE 2 : Configuration du routage ===\n");

        // Création des ContentRetrievers pour chaque source
        ContentRetriever retrieverRAG = createContentRetriever(embeddingStoreRAG, embeddingModel);
        ContentRetriever retrieverCyber = createContentRetriever(embeddingStoreCyber, embeddingModel);

        // Configuration des descriptions pour le routage intelligent
        Map<ContentRetriever, String> retrieverDescriptions = new HashMap<>();
        
        retrieverDescriptions.put(retrieverRAG,
                "Documents techniques sur l'intelligence artificielle, le RAG (Retrieval-Augmented Generation), " +
                "LangChain4j, les modèles de langage (LLM), les embeddings, les techniques avancées de RAG, " +
                "le machine learning, les agents IA, et les réseaux de neurones");
        
        retrieverDescriptions.put(retrieverCyber,
                "Documents sur la cybersécurité, la sécurité informatique, les menaces cyber, " +
                "la protection des données, les attaques informatiques, les pare-feu, " +
                "le chiffrement, et les bonnes pratiques de sécurité");

        System.out.println("Sources configurées :");
        System.out.println("  1. Support RAG/IA - " + documentRAG.getFileName());
        System.out.println("  2. Introduction à la cybersécurité - " + documentCyber.getFileName());

        // Création du QueryRouter qui utilisera le LLM pour choisir la bonne source
        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, retrieverDescriptions);
        System.out.println("\nQueryRouter créé avec le modèle " + modelName);

        // Configuration du RetrievalAugmentor avec le routeur
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // Création de l'assistant avec le routage intelligent
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant RAG avec routage intelligent prêt !\n");

        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print("Votre question : ");
            String question = scanner.nextLine().trim();

            if (question.equalsIgnoreCase("quitter") || question.equalsIgnoreCase("exit")) {
                System.out.println("\nAu revoir !");
                break;
            }

            if (question.isEmpty()) {
                System.out.println("Veuillez poser une question.\n");
                continue;
            }

            try {
                System.out.println("\n[Routage en cours - Consultez les logs pour voir la décision du LLM]");
                System.out.println("=".repeat(70));
                String reponse = assistant.chat(question);
                System.out.println("=".repeat(70));
                System.out.println("\nRéponse : " + reponse + "\n");
            } catch (Exception e) {
                System.err.println("Erreur : " + e.getMessage());
                e.printStackTrace();
            }
        }
        scanner.close();
    }
}
