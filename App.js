import "./App.css";
import { useEffect, useState } from "react";

function App() {
  const [frameSrc, setFrameSrc] = useState("");

  useEffect(() => {
    const eventSource = new EventSource(
      "http://localhost:8000/video_feed"
    );

    eventSource.addEventListener("NO_FRAMES", (event) => {
      console.log("frame not available");
    });

    eventSource.onmessage = (event) => {
      const eventData = event;

      if (
        eventData.event === "NO_FRAMES" ||
        eventData.event === "STREAM_INTERRUPTED"
      ) {
        console.log("Some Error.");
      } else {
        setFrameSrc(eventData.data);
      }
    };

    return () => {
      eventSource.close();
    };
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <img
          height={"200px"}
          width={"400px"}
          src={frameSrc}
          alt="Video Stream"
        />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
