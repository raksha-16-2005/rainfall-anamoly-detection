import express from 'express';
import mongoose from 'mongoose';
import Alert from '../models/Alert.js';

const router = express.Router();

/**
 * GET /api/alerts
 * Returns paginated alerts with filters and sorting
 *
 * Query params:
 * - risk_level: Filter by risk level (High Risk or Critical Risk)
 * - acknowledged: Filter by acknowledged status ('true' or 'false')
 * - page: Page number (default 1)
 * - limit: Results per page (default 20)
 *
 * Sorting: Critical Risk first, then High Risk, then by date descending
 */
router.get('/', async (req, res, next) => {
  try {
    const { risk_level, acknowledged, page = 1, limit = 20 } = req.query;
    const pageNum = Math.max(1, parseInt(page) || 1);
    const limitNum = Math.max(1, parseInt(limit) || 20);
    const skip = (pageNum - 1) * limitNum;

    // Build filter
    const filter = {};
    if (risk_level) {
      filter.risk_level = risk_level;
    }
    if (acknowledged !== undefined) {
      filter.acknowledged = acknowledged === 'true';
    }

    // Aggregation pipeline with severity sorting
    const pipeline = [
      { $match: filter },
      {
        $addFields: {
          severity: {
            $switch: {
              branches: [
                { case: { $eq: ['$risk_level', 'Critical Risk'] }, then: 2 },
                { case: { $eq: ['$risk_level', 'High Risk'] }, then: 1 },
              ],
              default: 0,
            },
          },
        },
      },
      {
        $sort: { severity: -1, date: -1 },
      },
    ];

    // Get total count
    const totalPipeline = [...pipeline, { $count: 'total' }];
    const countResult = await Alert.aggregate(totalPipeline);
    const total = countResult.length > 0 ? countResult[0].total : 0;

    // Get paginated results
    const alerts = await Alert.aggregate([
      ...pipeline,
      { $skip: skip },
      { $limit: limitNum },
      {
        $project: {
          severity: 0, // Remove the temporary severity field
        },
      },
    ]);

    const pages = Math.ceil(total / limitNum);

    res.json({ alerts, total, page: pageNum, pages });
  } catch (error) {
    next(error);
  }
});

/**
 * PATCH /api/alerts/:id/acknowledge
 * Marks an alert as acknowledged
 *
 * Sets acknowledged=true and acknowledged_at to current timestamp.
 * Returns the updated alert document.
 */
router.patch('/:id/acknowledge', async (req, res, next) => {
  try {
    const { id } = req.params;

    // Validate ObjectId
    if (!mongoose.Types.ObjectId.isValid(id)) {
      const error = new Error('Alert not found');
      error.status = 404;
      return next(error);
    }

    const updatedAlert = await Alert.findByIdAndUpdate(
      id,
      {
        acknowledged: true,
        acknowledged_at: new Date(),
      },
      { new: true }
    );

    if (!updatedAlert) {
      const error = new Error('Alert not found');
      error.status = 404;
      return next(error);
    }

    res.json(updatedAlert);
  } catch (error) {
    next(error);
  }
});

export default router;
