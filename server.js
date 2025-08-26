const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware for JSON parsing
app.use(express.json());

// Serve static files from current directory
app.use(express.static(path.join(__dirname)));

// API endpoint to generate crossword with Python
app.post('/api/generate-python', (req, res) => {
    console.log('🐍 Generating crossword with Python...');
    
    const python = spawn('python3', ['simple_crossword.py'], {
        cwd: __dirname
    });
    
    let stdout = '';
    let stderr = '';
    
    python.stdout.on('data', (data) => {
        stdout += data.toString();
    });
    
    python.stderr.on('data', (data) => {
        stderr += data.toString();
    });
    
    python.on('close', (code) => {
        console.log(`Python generator exited with code ${code}`);
        
        if (code === 0) {
            // Try to read the generated file
            const possibleFiles = [
                'simple_crossword.html',
                'cruciverba.html',
                'crossword.html'
            ];
            
            for (const filename of possibleFiles) {
                if (fs.existsSync(path.join(__dirname, filename))) {
                    fs.readFile(path.join(__dirname, filename), 'utf8', (err, data) => {
                        if (err) {
                            console.error('Error reading generated file:', err);
                            res.status(500).json({ 
                                error: 'Failed to read generated file',
                                details: err.message 
                            });
                        } else {
                            console.log('✅ Successfully generated and read crossword file');
                            res.json({ 
                                success: true, 
                                html: data,
                                filename: filename 
                            });
                        }
                    });
                    return;
                }
            }
            
            // No file found
            res.status(500).json({ 
                error: 'No generated crossword file found',
                stdout: stdout,
                stderr: stderr
            });
        } else {
            console.error('Python generator failed:', stderr);
            res.status(500).json({ 
                error: 'Python generator failed',
                code: code,
                stdout: stdout,
                stderr: stderr
            });
        }
    });
    
    python.on('error', (err) => {
        console.error('Failed to start Python process:', err);
        res.status(500).json({ 
            error: 'Failed to start Python generator',
            details: err.message 
        });
    });
});

// API endpoint to get generated files list
app.get('/api/files', (req, res) => {
    try {
        const files = fs.readdirSync(__dirname)
            .filter(file => file.endsWith('.html') && file !== 'index.html')
            .map(file => ({
                name: file,
                path: `/${file}`,
                size: fs.statSync(path.join(__dirname, file)).size,
                modified: fs.statSync(path.join(__dirname, file)).mtime
            }));
        
        res.json({ files });
    } catch (error) {
        res.status(500).json({ 
            error: 'Failed to list files',
            details: error.message 
        });
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    const healthStatus = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        version: process.version,
        platform: process.platform
    };
    
    res.json(healthStatus);
});

// API info endpoint
app.get('/api/info', (req, res) => {
    res.json({
        name: 'Crossword Generator',
        version: '1.0.0',
        description: 'Modern crossword generator with enhanced UI/UX',
        endpoints: [
            'GET /health - Health check',
            'GET /api/info - API information',
            'GET /api/files - List generated files',
            'POST /api/generate-python - Generate crossword with Python'
        ],
        features: [
            'Modern responsive UI',
            'Multiple grid sizes',
            'Theme-based generation',
            'Timer and progress tracking',
            'Hint system',
            'Solution validation',
            'Export functionality'
        ]
    });
});

// Root endpoint - serve main interface
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Catch-all handler for SPA routing
app.get('*', (req, res) => {
    // If it's an API request, return 404
    if (req.path.startsWith('/api/')) {
        return res.status(404).json({ error: 'API endpoint not found' });
    }
    
    // For other requests, try to serve the file or fallback to index.html
    const filePath = path.join(__dirname, req.path);
    if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
        res.sendFile(filePath);
    } else {
        res.sendFile(path.join(__dirname, 'index.html'));
    }
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('Server error:', error);
    res.status(500).json({ 
        error: 'Internal server error',
        details: process.env.NODE_ENV === 'development' ? error.message : undefined
    });
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('🛑 Received SIGTERM, shutting down gracefully');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('🛑 Received SIGINT, shutting down gracefully');
    process.exit(0);
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
    console.log('🧩 Crossword Generator Server Started');
    console.log('=====================================');
    console.log(`🌐 Server running on port ${PORT}`);
    console.log(`📱 Web interface: http://localhost:${PORT}`);
    console.log(`❤️  Health check: http://localhost:${PORT}/health`);
    console.log(`🔗 API info: http://localhost:${PORT}/api/info`);
    console.log(`📂 Environment: ${process.env.NODE_ENV || 'development'}`);
    console.log('=====================================');
});
