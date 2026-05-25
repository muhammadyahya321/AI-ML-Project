const { getBackendUrl } = require("./_backend");

module.exports = async (req, res) => {
  const backendUrl = getBackendUrl();

  try {
    const response = await fetch(new URL("/api/health", backendUrl), {
      headers: { Accept: "application/json" },
    });
    const text = await response.text();
    res.status(response.status);
    res.setHeader("Content-Type", response.headers.get("content-type") || "application/json");
    res.send(text);
  } catch (error) {
    res.status(502).json({ error: `Backend health request failed: ${error.message}` });
  }
};
