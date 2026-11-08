package com.example.ai.config;

import org.springframework.boot.web.client.RestClientCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.client.BufferingClientHttpRequestFactory;
import org.springframework.http.client.ClientHttpResponse;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.util.StreamUtils;

import java.nio.charset.StandardCharsets;

/**
 * Enterprise HTTP Logging Configuration for Spring AI & Ollama.
 * Intercepts the underlying RestClient communication between Spring Boot and local Ollama,
 * printing the raw HTTP Request and Response JSON payloads directly to the console.
 */
@Configuration
public class OllamaHttpLoggingConfig {

    @Bean
    public RestClientCustomizer ollamaLoggingCustomizer() {
        return restClientBuilder -> restClientBuilder
                .requestFactory(new BufferingClientHttpRequestFactory(new SimpleClientHttpRequestFactory()))
                .requestInterceptor((request, body, execution) -> {
                    // Only log requests targeting Ollama (port 11434)
                    boolean isOllama = request.getURI().toString().contains("11434");

                    if (isOllama) {
                        String reqBody = new String(body, StandardCharsets.UTF_8);
                        System.out.println("\n==================== [1. HTTP Request -> Ollama] ====================");
                        System.out.println("URL: " + request.getURI());
                        System.out.println(reqBody);
                    }

                    ClientHttpResponse response = execution.execute(request, body);

                    if (isOllama) {
                        String respBody = StreamUtils.copyToString(response.getBody(), StandardCharsets.UTF_8);
                        System.out.println("==================== [2. HTTP Response <- Ollama Raw JSON] ====================");
                        System.out.println(respBody);
                        System.out.println("=================================================================================\n");
                    }

                    return response;
                });
    }
}
