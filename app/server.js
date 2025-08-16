const express = require('express');
const axios = require('axios');
const helmet = require('helmet');
const cors = require('cors');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
const { body, validationResult } = require('express-validator');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const MCP_SERVER_URL = process.env.MCP_SERVER_URL || 'http://mcp-server:8001';

// Security middleware
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            styleSrc: ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
            scriptSrc: ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
            imgSrc: ["'self'", "data:", "https:"],
            connectSrc: ["'self'"],
            fontSrc: ["'self'", "https://cdn.jsdelivr.net"],
        },
    },
}));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP, please try again later.'
});
app.use(limiter);

// CORS
app.use(cors());

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Logging
app.use(morgan('combined'));

// Static files
app.use(express.static('public'));

// Health check
app.get('/health', (req, res) => {
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

// API Routes
app.post('/api/query', [
    body('question').isLength({ min: 1, max: 500 }).trim().escape()
], async (req, res) => {
    try {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }

        const { question } = req.body;
        
        console.log(`Processing query: ${question}`);
        
        const response = await axios.post(`${MCP_SERVER_URL}/query`, {
            question,
            context: {}
        }, {
            timeout: 30000
        });
        
        res.json(response.data);
        
    } catch (error) {
        console.error('Query error:', error.message);
        
        if (error.response) {
            res.status(error.response.status).json({
                error: error.response.data.detail || 'Server error'
            });
        } else if (error.code === 'ECONNREFUSED') {
            res.status(503).json({
                error: 'AI service is currently unavailable. Please try again later.'
            });
        } else {
            res.status(500).json({
                error: 'An unexpected error occurred. Please try again.'
            });
        }
    }
});

app.get('/api/schema', async (req, res) => {
    try {
        const response = await axios.get(`${MCP_SERVER_URL}/schema`, {
            timeout: 10000
        });
        
        res.json(response.data);
        
    } catch (error) {
        console.error('Schema error:', error.message);
        
        if (error.response) {
            res.status(error.response.status).json({
                error: error.response.data.detail || 'Server error'
            });
        } else {
            res.status(500).json({
                error: 'Could not retrieve database schema'
            });
        }
    }
});

// Serve main application
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ error: 'Something went wrong!' });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
    console.log(`AI Insights Web App running on port ${PORT}`);
});
