import React, { useState } from 'react';

function StreamLinkAndPreview() {
    const [output, setOutput] = useState(null);
    const [error, setError] = useState(null);
    const [rtspUrl, setRtspUrl] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);


    const handleRtspSubmit = async () => {
        if (isProcessing) {
            return;
        }
        setIsProcessing(true);

        const formData = new FormData();
        formData.append('rtspUrl', rtspUrl);
        const response = await fetch('http://localhost:5000/process_rtsp', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
            },
            body: formData
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
                <input
                    type="text"
                    value={rtspUrl}
                    onChange={(e) => setRtspUrl(e.target.value)}
                    placeholder="Enter RTSP stream URL"
                    className='rtsp-input'
                />
                <button className='rtsp-button' onClick={handleRtspSubmit}>Process RTSP</button>
            </div>
            <div className='preview-box'>
                {isProcessing ? 'Processing' : output ? (
                    <img className='rtsp_imgstream' src={output} alt='rtspstream' />
                ) : (error ? error : (
                    'No file to process.'
                ))}
            </div>
        </div>
    );
}

export default StreamLinkAndPreview;