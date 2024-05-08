import React, { useState } from 'react';
import './App.css';
import FileDropAndPreview from './Components/FileDropAndPreview';
import StreamLinkAndPreview from './Components/StreamLinkAndPreview';

function App() {
  const [output, setOutput] = useState(null);
  const [mediaType, setMediaType] = useState('image');
  const [rstpUrl, setRstpUrl] = useState('');



  const handleWebcamProcessing = async () => {
    const response = await fetch('http://localhost:5000/process_webcam', {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
      },
    });

    if (response.ok) {
      const data = await response.json();
      setOutput(data.url);
    } else {
      setOutput(null);
    }
  };

  const handleMediaTypeChange = (e) => {
    console.log(e.target.value);
    setMediaType(e.target.value);
  };

  return (
    <div className="App">
      <h1>Fall Detection</h1>
      <form id="upload-form">

        <div className="media-type">
          <input id='image/video' type="radio" className='peer' name="mediaType" value="image/video" checked={mediaType === 'image/video'} onChange={handleMediaTypeChange} />
          <label htmlFor="image/video" height={'5rem'} >
            <img className='icon' src="/media.png" alt='media' height={'100%'} />
            <span>Image / Video</span>
          </label>

          <input id="rtsp" type="radio" className='peer' name="mediaType" value="rtsp" checked={mediaType === 'rtsp'} onChange={handleMediaTypeChange} />
          <label htmlFor="rtsp">
            <img className='icon' src="/rtsp.png" alt='rtsp' height={'100%'} />
            <span>RTSP Stream</span>
          </label>

          <input id='webcam' type="radio" className='peer' name="mediaType" value="webcam" checked={mediaType === 'webcam'} onChange={handleMediaTypeChange} />
          <label htmlFor="webcam">
            <img src="/webcam.png" alt='webcam' height={'100%'} />
            <span>Webcam</span>
          </label>
        </div>

      </form>
      {mediaType === 'rtsp' && (
        <StreamLinkAndPreview />
      )}
      {mediaType === 'webcam' && (
        <button onClick={handleWebcamProcessing}>Process Webcam</button>
      )}
      {mediaType === 'image/video' && (
        <FileDropAndPreview />
      )}
    </div>
  );
}

export default App;