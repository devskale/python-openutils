# Test-Suite — Domänenkarte

80 Dateien, ~820 Tests, komplett hermetisch (10 s Laufzeit). Navigation nach
Domäne; Verhaltenstiefe statt Existenz-Checks ist die Norm (Kontrakt-Tests
sind parametrisiert, keine „method_exists"-Einzelprüfungen mehr).

| Domäne | Dateien (Auswahl) | Deckt |
|---|---|---|
| **Dispatch & Zuverlässigkeit** | test_completion_target, test_auto_router(+_e2e), test_stream_guard, test_retry_after, test_reliability, test_prime_timeout, test_stream_close_race | Target-Naht, Auto-Routing+Fallback, Empty-Guard, 429/Retry-After, Stream-Schutz |
| **Oberflächen** | test_responses_api, test_systemone_router, test_proxy_tu, test_proxy_async, test_ws_stt_stream, test_surface_smoke | Chat/Responses/WS-STT/Systemone über HTTP inkl. SSE-Envelope |
| **SSE & Protokoll** | test_sse, test_llms_txt, test_discovery, test_models_namespacing, test_openai_passthrough | Decoder-Seam, Agent-Discovery, alias@model |
| **Auth & Metering** | test_auth, test_key_usage, test_token_registry, test_instance_auth, test_auth_ban, test_proxy_auth_gate | Tokens, Allowlist, X-Provider-Key, Stats |
| **Provider-Verhalten** | test_tu_* (8), test_opencode_provider, test_gemini_*, test_glm_smoke, test_kilo/pollinations/mistral/role_developer | Transport-Stalls, Reasoning-Dialekte, Leak-Repair, Quota (AIP-193) |
| **Provider-Kontrakt** | test_async_providers | parametrisiert: Interface-Shapes aller ChatProvider |
| **Instanz-Ebene** | test_instances, test_instance_*, test_alias_refresh, test_multimodal_instances | Overlays, Pool-Keys, enabled-Filter, Lazy-Refresh |
| **Katalog & Infra** | test_generate_models_routine, test_update_models_watchdog, test_watchdog_memguard, test_capabilities_probes, test_health_index, test_stats_ttft | Refresh, Watchdog, Probes |
| **Audio** | test_tts_stt_async, test_tts_integration, test_tts_streaming, test_audio_instance_alias, test_openai_compat_audio_typing | TTS/STT-Seams |

Historie: 2026-11-21 bereinigt — 7 Legacy-Nicht-Test-Dateien entfernt
(testEmbeddings lief als 3 ERRORs in jeder Suite mit), 34+9 flache
Existenz-Checks zu parametrisierten Kontrakt-Tests verdichtet.
