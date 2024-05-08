import React, { useState } from 'react';

function StreamLinkAndPreview() {
    const [output, setOutput] = useState(null);
    const [error, setError] = useState(null);
    const [rtspUrl, setRtspUrl] = useState('');


    const handleRtspSubmit = async () => {
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
            setOutput(data.url);
            setError(null);
        } else {
            setOutput(null);
            setError(data.error)
        }
    };

    return (
        <div className='container'>
            <div>
                <input
                    type="text"
                    value={rtspUrl}
                    onChange={(e) => setRtspUrl(e.target.value)}
                    placeholder="Enter RTSP stream URL"
                />
                <button onClick={handleRtspSubmit}>Process RTSP</button>
            </div>
            <div>
                {output ? (
                    <img className='rtsp_imgstream' src={output} alt='rtspstream' />
                ) : (error ? error : (
                    'No file to process.'
                ))}
            </div>
        </div>
    );
}

export default StreamLinkAndPreview;