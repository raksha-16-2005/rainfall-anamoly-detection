import mongoose from 'mongoose';

const PredictionSchema = new mongoose.Schema({
  district: {
    type: String,
    required: true,
    lowercase: true,
    trim: true,
  },
  date: {
    type: Date,
    required: true,
  },
  rainfall_avg: {
    type: Number,
    default: null,
  },
  rainfall_gov: {
    type: Number,
    default: null,
  },
  rainfall_meteo: {
    type: Number,
    default: null,
  },
  dual_source: {
    type: Boolean,
    default: null,
  },
  predicted_mm: {
    type: Number,
    default: null,
  },
  yhat_lower: {
    type: Number,
    default: null,
  },
  yhat_upper: {
    type: Number,
    default: null,
  },
  anomaly_flag: {
    type: Boolean,
    default: null,
  },
  anomaly_score: {
    type: Number,
    default: null,
  },
  in_regional_cluster: {
    type: Boolean,
    default: null,
  },
  rolling_zscore: {
    type: Number,
    default: null,
  },
  zscore_category: {
    type: String,
    enum: ['normal', 'moderate', 'extreme'],
    default: null,
  },
  risk_level: {
    type: String,
    enum: ['Normal', 'Moderate Risk', 'High Risk', 'Critical Risk'],
    required: true,
  },
  confidence: {
    type: String,
    default: null,
  },
  reason: {
    type: String,
    default: null,
  },
  created_at: {
    type: Date,
    default: Date.now,
  },
});

// Compound unique index on (district, date)
PredictionSchema.index({ district: 1, date: 1 }, { unique: true });

// Single-field index on risk_level
PredictionSchema.index({ risk_level: 1 });

export default mongoose.model('Prediction', PredictionSchema);
