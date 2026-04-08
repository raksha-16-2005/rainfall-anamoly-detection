import express from 'express';
import { Parser } from 'json2csv';
import Prediction from '../models/Prediction.js';

const router = express.Router();

/**
 * GET /api/export/csv
 * Exports predictions as CSV with optional filtering
 *
 * Query params:
 * - start_date: ISO string (filters date >= start_date)
 * - end_date: ISO string (filters date <= end_date)
 * - risk_level: Filter by risk level
 *
 * Returns CSV file with fields: district, date, rainfall_avg, predicted_mm,
 * anomaly_flag, rolling_zscore, zscore_category, risk_level, confidence, reason
 */
router.get('/csv', async (req, res, next) => {
  try {
    const { start_date, end_date, risk_level } = req.query;

    // Build filter
    const filter = {};

    // Date range filtering
    if (start_date || end_date) {
      filter.date = {};
      if (start_date) {
        filter.date.$gte = new Date(start_date);
      }
      if (end_date) {
        filter.date.$lte = new Date(end_date);
      }
    }

    // Risk level filtering
    if (risk_level) {
      filter.risk_level = risk_level;
    }

    // Query predictions with selected fields using .lean() for performance
    const predictions = await Prediction.find(filter)
      .select(
        'district date rainfall_avg predicted_mm anomaly_flag rolling_zscore zscore_category risk_level confidence reason'
      )
      .lean();

    // If no results, return 204 No Content
    if (predictions.length === 0) {
      return res.status(204).send();
    }

    // Define CSV fields
    const csvFields = [
      'district',
      'date',
      'rainfall_avg',
      'predicted_mm',
      'anomaly_flag',
      'rolling_zscore',
      'zscore_category',
      'risk_level',
      'confidence',
      'reason',
    ];

    // Parse data to CSV
    const json2csvParser = new Parser({ fields: csvFields });
    const csv = json2csvParser.parse(predictions);

    // Generate filename with today's date (YYYY-MM-DD)
    const today = new Date();
    const dateStr = today.toISOString().split('T')[0];
    const filename = `rainfall_report_${dateStr}.csv`;

    // Set response headers
    res.set({
      'Content-Type': 'text/csv',
      'Content-Disposition': `attachment; filename="${filename}"`,
    });

    res.send(csv);
  } catch (error) {
    next(error);
  }
});

export default router;
