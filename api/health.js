module.exports = async (req, res) => {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    res.status(500).json({ error: "BACKEND_URL is not configured on Vercel." });
    return;
  }

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
