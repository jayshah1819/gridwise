const isLocalhost =
  window.location.hostname === "localhost" ||
  window.location.hostname === "127.0.0.1" ||
  window.location.hostname.startsWith("192.168.") || // Local network
  window.location.hostname.startsWith("10.") ||      // Private network  
  window.location.hostname.startsWith("172.") ||     // Private network
  window.location.protocol === "file:";

const cacheBust = Date.now(); // Force cache refresh
const modulePath =
  (isLocalhost ? `/benchmarking.mjs?v=${cacheBust}` : "https://gridwise-webgpu.github.io/gridwise/benchmarking.mjs");

import(modulePath)
  .then(({ main }) => {
    main(navigator);
  })
  .catch((error) => {
    console.error("Error loading module", error);
  });

//import { main } from "http://localhost:8000/gridwise/benchmarking.mjs";
// main(navigator);
