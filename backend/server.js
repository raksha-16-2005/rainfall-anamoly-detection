import express from 'express';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI;

/**
 * Configure middleware
 */
app.use(cors({
  origin: 'http://localhost:5173',
  credentials: true,
}));

app.use(express.json());

/**
 * Connect to MongoDB Atlas
 */
async function connectDatabase() {
  try {
    if (!MONGO_URI) {
      throw new Error('MONGO_URI environment variable is not set');
    }

    await mongoose.connect(MONGO_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });

    console.log('MongoDB connected');
  } catch (error) {
    console.error('MongoDB connection error:', error.message);
    process.exit(1);
  }
}

/**
 * Health check endpoint
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date(),
  });
});

/**
 * Mount routers
 */
import districtsRouter from './routes/districts.js';
import alertsRouter from './routes/alerts.js';
import exportRouter from './routes/export.js';

app.use('/api/districts', districtsRouter);
app.use('/api/alerts', alertsRouter);
app.use('/api/export', exportRouter);

/**
 * Centralized error handler middleware
 */
app.use((err, req, res, next) => {
  const status = err.status || 500;
  const message = err.message || 'Internal server error';

  console.error(`[Error] Status: ${status}, Message: ${message}`);

  res.status(status).json({
    error: message,
  });
});

/**
 * Start server
 */
async function startServer() {
  await connectDatabase();

  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
  });
}

startServer().catch((error) => {
  console.error('Failed to start server:', error.message);
  process.exit(1);
});
