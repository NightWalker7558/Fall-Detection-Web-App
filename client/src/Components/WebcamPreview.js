import React, { useState } from 'react';

function WebcamPreview() {
    const [output, setOutput] = useState(null);
    const [error, setError] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);


    const handleWebcamProcessing = async () => {
        if (isProcessing) {
            return;
        }
        setIsProcessing(true);
        const response = await fetch('http://localhost:5000/process_webcam', {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
            },
        });

        const data = await response.json();

        if (response.ok) {
            setOutput(data.url + '?t=' + Date.now());
            setError(null);
        } else {
            setOutput(null);
            setError(data.error)
        }
        setIsProcessing(false);
    };

    return (
        <div className='container'>
            <div>
                <button className='webcam-button' onClick={handleWebcamProcessing}>Process Webcam</button>
            </div>
            <div className='preview-box'>
                {isProcessing ? 'Processing...' : output ? (
                    <img className='webcam' src={output} alt='webcam' />
                ) : (error ? error : (
                    'No file to process.'
                ))}
            </div>
        </div>
    );
}

export default WebcamPreview;