import uuid
from collections import OrderedDict

from flask import Flask, render_template, request, send_file, abort, url_for
from analysis import AnalysisError, build_report, png_to_data_uri
from pdf_report import build_pdf
import io

app = Flask(__name__)

# Simple in-memory cache so the PDF download doesn't require re-fetching data.
# Bounded so a long-running local session doesn't grow unbounded.
REPORTS = OrderedDict()
MAX_CACHED_REPORTS = 25


def _cache_put(report_id, report):
    REPORTS[report_id] = report
    while len(REPORTS) > MAX_CACHED_REPORTS:
        REPORTS.popitem(last=False)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    query = request.form.get("company", "")
    try:
        report = build_report(query)
    except AnalysisError as e:
        return render_template("index.html", error=str(e), query=query), 400
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Something went wrong fetching or analyzing this company: {e}",
            query=query,
        ), 500

    report_id = uuid.uuid4().hex
    _cache_put(report_id, report)

    images_b64 = {k: png_to_data_uri(v) for k, v in report["images"].items()}
    return render_template("report.html", r=report, images=images_b64, report_id=report_id)


@app.route("/report/<report_id>/download")
def download(report_id):
    report = REPORTS.get(report_id)
    if report is None:
        abort(404, "This report has expired. Please run the analysis again.")
    pdf_bytes = build_pdf(report)
    filename = f"{report['ticker'].replace('.', '_')}_vs_nifty50_report.pdf"
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8003, debug=False)
