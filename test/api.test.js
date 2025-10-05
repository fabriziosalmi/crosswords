/**
 * Basic API Tests for Crossword Generator
 * Simple tests using Node.js built-in modules
 */

const http = require('http');
const { spawn } = require('child_process');
const path = require('path');

// Test configuration
const PORT = process.env.TEST_PORT || 3001;
const HOST = 'localhost';
let serverProcess;

// Helper function to make HTTP requests
function makeRequest(path, method = 'GET', data = null) {
    return new Promise((resolve, reject) => {
        const options = {
            hostname: HOST,
            port: PORT,
            path: path,
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const req = http.request(options, (res) => {
            let body = '';
            
            res.on('data', (chunk) => {
                body += chunk;
            });
            
            res.on('end', () => {
                try {
                    const result = {
                        statusCode: res.statusCode,
                        headers: res.headers,
                        body: body,
                        json: null
                    };
                    
                    if (res.headers['content-type']?.includes('application/json')) {
                        result.json = JSON.parse(body);
                    }
                    
                    resolve(result);
                } catch (e) {
                    reject(e);
                }
            });
        });

        req.on('error', reject);
        
        if (data) {
            req.write(JSON.stringify(data));
        }
        
        req.end();
    });
}

// Start test server
function startServer() {
    return new Promise((resolve, reject) => {
        serverProcess = spawn('node', ['server.js'], {
            cwd: path.join(__dirname, '..'),
            env: { ...process.env, PORT: PORT }
        });

        serverProcess.stdout.on('data', (data) => {
            if (data.toString().includes('Server running')) {
                setTimeout(resolve, 1000); // Wait a bit more for server to be ready
            }
        });

        serverProcess.stderr.on('data', (data) => {
            console.error(`Server error: ${data}`);
        });

        serverProcess.on('error', reject);

        // Timeout if server doesn't start
        setTimeout(() => reject(new Error('Server start timeout')), 10000);
    });
}

// Stop test server
function stopServer() {
    return new Promise((resolve) => {
        if (serverProcess) {
            serverProcess.on('exit', resolve);
            serverProcess.kill('SIGTERM');
            setTimeout(() => {
                if (serverProcess) {
                    serverProcess.kill('SIGKILL');
                    resolve();
                }
            }, 5000);
        } else {
            resolve();
        }
    });
}

// Test runner
async function runTests() {
    const tests = [];
    let passed = 0;
    let failed = 0;

    console.log('🧪 Starting API Tests...\n');

    try {
        // Start server
        console.log('🚀 Starting test server...');
        await startServer();
        console.log('✅ Test server started on port', PORT);
        console.log('');

        // Test 1: Health endpoint
        tests.push({
            name: 'GET /health should return healthy status',
            test: async () => {
                const response = await makeRequest('/health');
                if (response.statusCode !== 200) {
                    throw new Error(`Expected status 200, got ${response.statusCode}`);
                }
                if (!response.json || response.json.status !== 'healthy') {
                    throw new Error('Health check failed');
                }
                return true;
            }
        });

        // Test 2: API info endpoint
        tests.push({
            name: 'GET /api/info should return API information',
            test: async () => {
                const response = await makeRequest('/api/info');
                if (response.statusCode !== 200) {
                    throw new Error(`Expected status 200, got ${response.statusCode}`);
                }
                if (!response.json || !response.json.name) {
                    throw new Error('API info missing name field');
                }
                if (!response.json.endpoints || !Array.isArray(response.json.endpoints)) {
                    throw new Error('API info missing endpoints array');
                }
                return true;
            }
        });

        // Test 3: Files endpoint
        tests.push({
            name: 'GET /api/files should return files list',
            test: async () => {
                const response = await makeRequest('/api/files');
                if (response.statusCode !== 200) {
                    throw new Error(`Expected status 200, got ${response.statusCode}`);
                }
                if (!response.json || !Array.isArray(response.json.files)) {
                    throw new Error('Files endpoint should return files array');
                }
                return true;
            }
        });

        // Test 4: Root endpoint
        tests.push({
            name: 'GET / should return index.html',
            test: async () => {
                const response = await makeRequest('/');
                if (response.statusCode !== 200) {
                    throw new Error(`Expected status 200, got ${response.statusCode}`);
                }
                if (!response.body.includes('Generatore di Cruciverba')) {
                    throw new Error('Index page does not contain expected title');
                }
                return true;
            }
        });

        // Test 5: 404 for non-existent API endpoint
        tests.push({
            name: 'GET /api/nonexistent should return 404',
            test: async () => {
                const response = await makeRequest('/api/nonexistent');
                if (response.statusCode !== 404) {
                    throw new Error(`Expected status 404, got ${response.statusCode}`);
                }
                return true;
            }
        });

        // Run all tests
        for (const test of tests) {
            try {
                await test.test();
                console.log(`✅ ${test.name}`);
                passed++;
            } catch (error) {
                console.log(`❌ ${test.name}`);
                console.log(`   Error: ${error.message}`);
                failed++;
            }
        }

    } catch (error) {
        console.error('❌ Test setup error:', error.message);
        process.exit(1);
    } finally {
        // Stop server
        console.log('\n🛑 Stopping test server...');
        await stopServer();
        console.log('✅ Test server stopped');
    }

    // Print summary
    console.log('\n📊 Test Summary');
    console.log('================');
    console.log(`Total: ${tests.length}`);
    console.log(`✅ Passed: ${passed}`);
    console.log(`❌ Failed: ${failed}`);
    console.log('');

    // Exit with appropriate code
    process.exit(failed > 0 ? 1 : 0);
}

// Run tests if this file is executed directly
if (require.main === module) {
    runTests().catch((error) => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

module.exports = { runTests, makeRequest };
