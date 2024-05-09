import React, { useState, useRef } from 'react';

function FileDropAndPreview() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [output, setOutput] = useState(null);
  const [drag, setDrag] = useState(false);
  const [dropText, setDropText] = useState('Drag and drop files here');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
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
    if (isProcessing) {
      return;
    }
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('file', file);
    const endpoint = file.type.startsWith('image/') ? 'http://localhost:5000/process_image' : 'http://localhost:5000/process_video';
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Accept': 'application/json',
      },
      body: formData
    })

    const data = await response.json();

    if (response.ok) {

      setOutput(data.url);
      setError(null);
    } else {
      setOutput(null);
      setError(data.error);
    }

    setIsProcessing(false);
  };

  const handleDropAreaClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div>
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

          {isProcessing ? 'Processing' : (
            output ? (
              file && file.type.startsWith('image/') ? (
                <img src={output} alt="output" />
              ) : (
                <video controls>
                  <source src={output} type="video/mp4" />
                  <source src={output} type="video/ogg" />
                  <source src={output} type="video/webm" />
                  <source src={output} type="video/mov" />
                  <source src={output} type="video/avi" />
                  Your browser does not support the video tag.
                </video>
              )
            ) : error ? error : (
              'No file to process.'
            )
          )}
        </div>
      </div>
    </div>
  );
}

export default FileDropAndPreview;