from flask import Flask, request, jsonify
from flask_cors import CORS  # 🔌 تم إضافة موديول التواصل عبر النطاقات لفك حظر المتصفح
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import RegressionModel
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)
CORS(app)  # 🌐 تفعيل السيرفر لاستقبال الطلبات القادمة من المتصفحات محلياً أو سحابياً بشكل مباشر وآمن

def execute_dual_path_engine(m_list, forecast_days=7):
    total_data = np.array(m_list, dtype=float)
    total_n = len(total_data)
    
    # 🧠 التكتيك الذكي للبيانات الصغيرة جداً (لحماية السيرفر من الانهيار وتوليد تنبؤ ذكي)
    if total_n < 10:
        # إذا كانت القيمة واحدة أو قيمتين متطابقتين، نكررها، وإذا كانت قيم مختلفة نحسب الاتجاه (Trend)
        if total_n == 1:
            base_pred = total_data[0]
        else:
            # حساب التغير بين القيمة الأولى والأخيرة لمعرفة هل النفقات تصعد أم تهبط
            trend = (total_data[-1] - total_data[0]) / (total_n - 1)
            base_pred = total_data[-1] + trend
        
        # توليد مصفوفة التنبؤ بناءً على السلوك المكتشف وتقريبها لأقرب رقمين عشريين
        final_hybrid_forecast = [max(0.0, round(float(base_pred), 2))] * forecast_days
        return final_hybrid_forecast

    # 1. تهيئة التواريخ المتوافقة مع محرك Darts تلقائياً
    date_range = pd.date_range(start='2024-01-01', periods=total_n, freq='D')
    series_all = TimeSeries.from_series(pd.Series(total_data, index=date_range))
    
    # 🌟 المسار الأول: التحليل الاستراتيجي بعيد المدى (Macro Model)
    macro_lags = min(60, total_n - forecast_days - 1)
    # حماية إضافية لو كانت البيانات بين 10 و 35 يوم كي لا تصبح الـ lags صفر أو سالب
    if macro_lags <= 0: macro_lags = max(1, int(total_n * 0.2))
    
    macro_model = RegressionModel(
        model=RandomForestRegressor(n_estimators=150, random_state=42),
        lags=macro_lags, 
        output_chunk_length=forecast_days
    )
    macro_model.fit(series_all)
    macro_pred = macro_model.predict(n=forecast_days, series=series_all).values().flatten()
    
    # 🌟 المسار الثاني: التحليل التكتيكي قصير المدى (Micro Model)
    # اقتطاع ذكي: يأخذ آخر 30 يوماً أو كامل البيانات المتاحة إذا كانت أقل من 30
    micro_series = series_all[-30:]
    micro_lags = min(7, len(micro_series) - forecast_days - 1)
    if micro_lags <= 0: micro_lags = max(1, int(len(micro_series) * 0.2))

    micro_model = RegressionModel(
        model=RandomForestRegressor(n_estimators=100, random_state=42),
        lags=micro_lags,  # تعديل مرن تلقائي بدلاً من تثبيتها على 7 لتجنب الانهيار
        output_chunk_length=forecast_days
    )
    micro_model.fit(micro_series)
    micro_pred = micro_model.predict(n=forecast_days, series=micro_series).values().flatten()
    
    # 🌟 طبقة الدمج الهجين الموزون
    final_hybrid_forecast = (0.60 * macro_pred) + (0.40 * micro_pred)
    
    return np.round(final_hybrid_forecast, 2).tolist()

# 📬 بوابة استقبال طلبات الهاتف (API Endpoint)
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        content = request.get_json()
        
        if not content or 'data' not in content:
            return jsonify({
                "status": "error",
                "message": "❌ خطأ في الإدخال: يجب إرسال حقل 'data' محتوياً على مصفوفة النفقات التاريخية."
            }), 400
            
        expenses_history = content['data']
        predict_days = content.get('predict_days', 7) 
        
        # 🛡️ تم تصحيح الشرط الخاطئ: الآن يمنع فقط المصفوفات الفارغة تماماً لمنع الأخطاء البرمجية
        if len(expenses_history) <= 0:
            return jsonify({
                "status": "error",
                "message": "❌ السلسلة فارغة، يرجى إرسال سجل يحتوي على قيمة واحدة على الأقل للتنبؤ."
            }), 400

        # تشغيل المحرك المزدوج
        forecast_results = execute_dual_path_engine(expenses_history, forecast_days=predict_days)
        
        return jsonify({
            "status": "success",
            "engine": "Darts Dual-Path (Adaptive Mode)",
            "input_total_days": len(expenses_history),
            "predicted_days": predict_days,
            "predictions": forecast_results
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"💥 حدث خطأ داخلي في السيرفر: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
