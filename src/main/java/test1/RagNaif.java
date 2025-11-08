package test1;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.File;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;
import java.util.Scanner;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;



import static dev.langchain4j.data.document.loader.FileSystemDocumentLoader.loadDocument;

public class RagNaif {
    
    public static void main(String[] args) throws URISyntaxException {
        // Création du ChatModel avec Gemini
        String apiKey = System.getenv("GEMINI_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("Missing GEMINI_KEY environment variable.");
            return;
        }

        String modelName = System.getenv("GEMINI_MODEL");
        if (modelName == null || modelName.isBlank()) {
            modelName = "gemini-2.5-flash";
        }

        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName(modelName)
                .temperature(0.7)
                .maxOutputTokens(1024)
                .timeout(Duration.ofSeconds(120))
                .build();
        
        // ===== PHASE 1 : Enregistrement des embeddings =====
        
        // 1. Récupération du Path du fichier PDF
        URL resourceUrl = RagNaif.class.getResource("/support_rag.pdf");
        if (resourceUrl == null) {
            System.err.println("Erreur : Le fichier support_rag.pdf n'a pas été trouvé dans src/main/resources/");
            System.err.println("Veuillez placer le fichier PDF dans le répertoire src/main/resources/");
            return;
        }
        Path filePath = Paths.get(resourceUrl.toURI());
        
        // 2. Création d'un parser pour PDF
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        
        // 3. Chargement du fichier PDF
        Document document = loadDocument(filePath, parser);
        
        // 4. Création d'un DocumentSplitter et découpage en segments
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        List<TextSegment> segments = splitter.split(document);
        
        System.out.println("Nombre de segments créés : " + segments.size());
        
        // 5. Création du modèle d'embedding
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        
        // 6. Création des embeddings pour les segments
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        
        // 7. Ajout dans le magasin d'embeddings
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        
        System.out.println("Embeddings enregistrés : " + embeddings.size());
        
        // ===== PHASE 2 : Utilisation des embeddings pour répondre aux questions =====
        
        // 8. Création du ContentRetriever
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
        
        // 9. Ajout d'une mémoire pour 10 messages
        MessageWindowChatMemory chatMemory = MessageWindowChatMemory.withMaxMessages(10);
        
        // 10. Création de l'assistant avec le ContentRetriever
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .contentRetriever(contentRetriever)
                .chatMemory(chatMemory)
                .build();
        
        // 11. Boucle pour poser plusieurs questions
        Scanner scanner = new Scanner(System.in);
        System.out.println("\n=== Assistant RAG Naïf ===");
        System.out.println("Posez vos questions sur le contenu du PDF (tapez 'exit' pour quitter)");
        
        while (true) {
            System.out.print("\nVotre question : ");
            String question = scanner.nextLine();
            
            if (question.equalsIgnoreCase("exit")) {
                System.out.println("Au revoir !");
                break;
            }
            
            if (question.trim().isEmpty()) {
                continue;
            }
            
            System.out.println("\nAssistant : " + assistant.chat(question));
        }
        
        scanner.close();
    }
}
