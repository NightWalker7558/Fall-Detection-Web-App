import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [output, setOutput] = useState(null);
  const [drag, setDrag] = useState(false);
  const [dropText, setDropText] = useState('Drag and drop files here');
  const [mediaType, setMediaType] = useState('image');
  const fileInputRef = useRef();

  const handleDragOver = (e) => {
    e.preventDefault();
    setDrag(true);
    setDropText('Drop files');
  };

  const handleDragLeave = () => {
    setDrag(false);
    setDropText('Drag and drop files here');
  };

  const handleFileInput = (e) => {
    const file = e.target.files[0];
    setFile(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
      handleFileUpload(file);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDrag(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      setFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
        handleFileUpload(file);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const endpoint = file.type.startsWith('image/') ? 'http://localhost:5000/process_image' : 'http://localhost:5000/process_video';
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
      },
      body: formData
    });

    // const blob = await response.blob();
    // const url = URL.createObjectURL(blob);

    // Now you can use the URL to set the output
    if (response.ok) {
      const data = await response.json();
      setOutput(data.url);
    } else {
      setOutput(null);
    }
  };

  const handleDropAreaClick = () => {
    fileInputRef.current.click();
  };

  const handleMediaTypeChange = (e) => {
    setMediaType(e.target.value);
  };

  return (
    <div className="App">
      <h1>Fall Detection</h1>
      <form id="upload-form">
        <input type="radio" name="mediaType" value="image" checked={mediaType === 'image'} onChange={handleMediaTypeChange} /> Image
        <input type="radio" name="mediaType" value="video" checked={mediaType === 'video'} onChange={handleMediaTypeChange} /> Video
        <input type="radio" name="mediaType" value="rstp" checked={mediaType === 'rstp'} onChange={handleMediaTypeChange} /> RSTP
      </form>
      <div
        id="drop-area"
        onClick={handleDropAreaClick}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onDragLeave={handleDragLeave}
        className={drag ? 'highlight' : ''}
      >
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileInput}
        />
        <div className="drop-icon">
          <i className="fa-light fa-file-upload"></i>
        </div>
        <div className="drop-text">{dropText}</div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', flexDirection: 'row' }}>
        <div id="preview" className="preview-box">
          {preview ? (
            file && file.type.startsWith('image/') ? (
              <img src={preview} alt="preview" />
            ) : (
              <video src={preview} controls />
            )
          ) : (
            'No file uploaded yet.'
          )}
        </div>
        <div id="output" className="preview-box">
          {output ? (
            file && file.type.startsWith('image/') ? (
              <img src={output} alt="output" />
            ) : (
              <video src={output} controls type="video/mp4"/>
            )
          ) : (
            'No file to process.'
          )}
        </div>
      </div>
    </div>
  );
}

export default App;