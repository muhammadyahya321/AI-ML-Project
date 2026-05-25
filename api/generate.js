const { getBackendUrl } = require("./_backend");

module.exports = async (req, res) => {
  const backendUrl = getBackendUrl();

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
