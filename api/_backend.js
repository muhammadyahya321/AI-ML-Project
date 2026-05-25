const DEFAULT_RENDER_SERVICE_NAME = "ai-ml-project-backend";

function normalizeBaseUrl(value) {
  if (!value) {
    return "";
  }

  return String(value).trim().replace(/\/+$/, "");
}

function getBackendUrl() {
  const configuredUrl = normalizeBaseUrl(process.env.BACKEND_URL);
  if (configuredUrl) {
    return configuredUrl;
  }

  const configuredServiceName = normalizeBaseUrl(process.env.RENDER_SERVICE_NAME);
  const serviceName = configuredServiceName || DEFAULT_RENDER_SERVICE_NAME;
  return `https://${serviceName}.onrender.com`;
}

module.exports = {
  getBackendUrl,
};
