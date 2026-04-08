import express from 'express';
import Prediction from '../models/Prediction.js';

const router = express.Router();

/**
 * GET /api/districts
 * Returns the most recent prediction for each district
 *
 * Uses aggregation to find the latest document per district.
 * Sorts by date descending, groups by district, and takes first document.
 */
router.get('/', async (req, res, next) => {
  try {
    const districts = await Prediction.aggregate([
      {
        $sort: { date: -1 },
      },
      {
        $group: {
          _id: '$district',
          district: { $first: '$district' },
          risk_level: { $first: '$risk_level' },
          confidence: { $first: '$confidence' },
          rainfall_avg: { $first: '$rainfall_avg' },
          rolling_zscore: { $first: '$rolling_zscore' },
          predicted_mm: { $first: '$predicted_mm' },
          yhat_lower: { $first: '$yhat_lower' },
          yhat_upper: { $first: '$yhat_upper' },
          date: { $first: '$date' },
        },
      },
    ]);

    res.json({ districts });
  } catch (error) {
    next(error);
  }
});

/**
 * GET /api/districts/:name/history
 * Returns the last 30 predictions for a specific district
 *
 * Filters by district (lowercase), sorts by date ascending,
 * and limits to 30 most recent documents.
 */
router.get('/:name/history', async (req, res, next) => {
  try {
    const { name } = req.params;
    const district = name.toLowerCase();

    const history = await Prediction.find({ district })
      .select(
        'date rainfall_avg predicted_mm yhat_lower yhat_upper rolling_zscore risk_level anomaly_flag'
      )
      .sort({ date: 1 })
      .limit(30);

    res.json({ district, history });
  } catch (error) {
    next(error);
  }
});

export default router;
