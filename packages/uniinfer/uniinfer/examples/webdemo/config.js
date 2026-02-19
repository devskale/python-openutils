// Provide a global config object that can be updated from the UI
window.uniinferConfig = {
  // Default to current origin (e.g. http://localhost:8123) if running locally,
  // otherwise fallback or user can configure.
  apiBaseUrl: window.location.origin,
  apiKey: "",
  maxTokens: 4000,
};
