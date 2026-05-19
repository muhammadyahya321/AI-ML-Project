module.exports = async (req, res) => {
  const backendUrl = process.env.BACKEND_URL;
  if (!backendUrl) {
    res.status(500).json({ error: "BACKEND_URL is not configured on Vercel." });
    return;
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    res.status(405).json({ error: "Method not allowed." });
    return;
  }

  try {
    const response = await fetch(new URL("/api/generate", backendUrl), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(req.body || {}),
    });
    const text = await response.text();
    res.status(response.status);
    res.setHeader("Content-Type", response.headers.get("content-type") || "application/json");
    res.send(text);
  } catch (error) {
    res.status(502).json({ error: `Backend generate request failed: ${error.message}` });
  }
};
