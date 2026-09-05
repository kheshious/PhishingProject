from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
import sys


sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "machine_learning"
    )
)


from database.db_connection import get_db_connection
from predict import predict_message
from predict_email import predict_email
from predict_url import predict_url


app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "change-this-to-a-long-random-secret-key"
)


def login_required(view_function):

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please log in to continue.",
                "warning"
            )

            return redirect(
                url_for("login")
            )

        return view_function(
            *args,
            **kwargs
        )

    return wrapped_view

def admin_required(view_function):

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please log in to continue.",
                "warning"
            )

            return redirect(
                url_for("login")
            )


        user_role = str(
            session.get(
                "role",
                ""
            )
        ).strip().lower()


        if user_role != "admin":

            flash(
                "You do not have permission to access the administrator area.",
                "error"
            )

            return redirect(
                url_for("home")
            )


        return view_function(
            *args,
            **kwargs
        )


    return wrapped_view

def user_required(f):

    @wraps(f)
    def decorated_function(
        *args,
        **kwargs
    ):

        # User must be logged in
        if "user_id" not in session:

            return redirect(
                url_for(
                    "login"
                )
            )

        if (
            str(
                session.get(
                    "role",
                    ""
                )
            )
            .strip()
            .lower()
            == "admin"
        ):

            return redirect(
                url_for(
                    "admin_scam_reports"
                )
            )

        return f(
            *args,
            **kwargs
        )

    return decorated_function

def save_detection(
    input_type,
    submitted_content,
    prediction_result,
):

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO detections
        (
            user_id,
            input_type,
            submitted_content,
            prediction,
            confidence_score,
            risk_level,
            explanation
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            session["user_id"],
            input_type,
            submitted_content,
            prediction_result["prediction"],
            float(
                prediction_result["confidence"]
            ),
            prediction_result["risk_level"],
            "; ".join(
                prediction_result.get(
                    "reasons",
                    []
                )
            ),
        )
    )

    conn.commit()
    cursor.close()
    conn.close()


def get_module_progress(user_id, module_id):

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Count total lessons in the module
    cursor.execute("""
        SELECT COUNT(*) AS total_lessons
        FROM learning_lessons
        WHERE module_id = %s
    """, (
        module_id,
    ))

    total_lessons = (
        cursor.fetchone()["total_lessons"]
        or 0
    )


    # Count completed lessons for this user
    cursor.execute("""
        SELECT COUNT(*) AS completed_lessons
        FROM user_lesson_progress ulp
        INNER JOIN learning_lessons ll
            ON ulp.lesson_id = ll.lesson_id
        WHERE ulp.user_id = %s
          AND ll.module_id = %s
          AND ulp.completed = 1
    """, (
        user_id,
        module_id,
    ))

    completed_lessons = (
        cursor.fetchone()["completed_lessons"]
        or 0
    )

    cursor.close()
    conn.close()


    if total_lessons > 0:

        progress_percentage = round(
            (
                completed_lessons
                / total_lessons
            ) * 100
        )

    else:

        progress_percentage = 0


    return {
        "total_lessons": total_lessons,
        "completed_lessons": completed_lessons,
        "progress_percentage": progress_percentage,
        "all_lessons_completed": (
            total_lessons > 0
            and
            completed_lessons == total_lessons
        )
    }

@app.route("/")
@user_required
def home():

    conn = get_db_connection()

    cursor = conn.cursor(
        dictionary=True
    )

    user_id = session["user_id"]

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
        """,
        (
            user_id,
        )
    )

    total_scans = (
        cursor.fetchone()["total"]
        or 0
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
          AND prediction IN (
              'Phishing',
              'Malicious'
          )
        """,
        (
            user_id,
        )
    )

    phishing_count = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
          AND prediction = 'Suspicious'
        """,
        (
            user_id,
        )
    )

    suspicious_count = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
          AND prediction = 'Legitimate'
        """,
        (
            user_id,
        )
    )

    legitimate_count = (
        cursor.fetchone()["total"]
        or 0
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
          AND risk_level = 'High'
        """,
        (
            user_id,
        )
    )

    high_risk_count = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
          AND risk_level = 'Medium'
        """,
        (
            user_id,
        )
    )

    medium_risk_count = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
          AND risk_level = 'Low'
        """,
        (
            user_id,
        )
    )

    low_risk_count = (
        cursor.fetchone()["total"]
        or 0
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE user_id = %s
        """,
        (
            user_id,
        )
    )

    scam_reports_count = (
        cursor.fetchone()["total"]
        or 0
    )

    cursor.execute(
        """
        SELECT
            ROUND(
                AVG(confidence_score),
                2
            ) AS avg_confidence
        FROM detections
        WHERE user_id = %s
        """,
        (
            user_id,
        )
    )

    avg_confidence = (
        cursor.fetchone()[
            "avg_confidence"
        ]
        or 0
    )

    cursor.execute(
        """
        SELECT
            input_type,
            prediction,
            confidence_score,
            risk_level,
            scan_date
        FROM detections
        WHERE user_id = %s
        ORDER BY scan_date DESC
        LIMIT 5
        """,
        (
            user_id,
        )
    )

    recent_scans = (
        cursor.fetchall()
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total_modules
        FROM learning_modules
        """
    )

    total_modules = (
        cursor.fetchone()[
            "total_modules"
        ]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS completed_modules
        FROM user_learning_progress
        WHERE user_id = %s
          AND module_completed = 1
        """,
        (
            user_id,
        )
    )

    modules_completed = (
        cursor.fetchone()[
            "completed_modules"
        ]
        or 0
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total_lessons
        FROM learning_lessons
        """
    )

    total_lessons = (
        cursor.fetchone()[
            "total_lessons"
        ]
        or 0
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS completed_lessons
        FROM user_lesson_progress ulp

        INNER JOIN learning_lessons ll
            ON ll.lesson_id = ulp.lesson_id

        WHERE ulp.user_id = %s
          AND ulp.completed = 1
        """,
        (
            user_id,
        )
    )

    lessons_completed = (
        cursor.fetchone()[
            "completed_lessons"
        ]
        or 0
    )


    if total_lessons > 0:

        learning_progress = round(
            (
                lessons_completed
                / total_lessons
            ) * 100
        )

    else:

        learning_progress = 0


    cursor.execute(
        """
        SELECT
            ROUND(
                AVG(best_quiz_score),
                2
            ) AS average_quiz_score
        FROM user_learning_progress
        WHERE user_id = %s
          AND best_quiz_score > 0
        """,
        (
            user_id,
        )
    )

    average_quiz_score = (
        cursor.fetchone()[
            "average_quiz_score"
        ]
        or 0
    )


    cursor.execute(
        """
        SELECT
            lm.module_id,
            lm.title,
            lm.category,
            lm.description,
            lm.icon,
            lm.display_order,

            COUNT(
                DISTINCT ll.lesson_id
            ) AS total_lessons,

            COUNT(
                DISTINCT CASE
                    WHEN ulp.completed = 1
                    THEN ll.lesson_id
                END
            ) AS completed_lessons,

            COALESCE(
                ulearn.best_quiz_score,
                0
            ) AS best_quiz_score,

            COALESCE(
                ulearn.module_completed,
                0
            ) AS module_completed

        FROM learning_modules lm

        LEFT JOIN learning_lessons ll
            ON ll.module_id = lm.module_id

        LEFT JOIN user_lesson_progress ulp
            ON ulp.lesson_id = ll.lesson_id
            AND ulp.user_id = %s

        LEFT JOIN user_learning_progress ulearn
            ON ulearn.module_id = lm.module_id
            AND ulearn.user_id = %s

        GROUP BY
            lm.module_id,
            lm.title,
            lm.category,
            lm.description,
            lm.icon,
            lm.display_order,
            ulearn.best_quiz_score,
            ulearn.module_completed

        ORDER BY
            lm.display_order
        """,
        (
            user_id,
            user_id,
        )
    )

    learning_modules = (
        cursor.fetchall()
    )

    for module in learning_modules:

        module_total = (
            module["total_lessons"]
            or 0
        )

        module_completed_lessons = (
            module["completed_lessons"]
            or 0
        )


        if module_total > 0:

            module["progress_percentage"] = round(
                (
                    module_completed_lessons
                    / module_total
                ) * 100
            )

        else:

            module["progress_percentage"] = 0


    continue_module = None


    for module in learning_modules:

        if (
            module["completed_lessons"] > 0
            and not module["module_completed"]
        ):

            continue_module = module

            break

    if continue_module is None:

        for module in learning_modules:

            if (
                module["completed_lessons"] == 0
                and not module["module_completed"]
            ):

                continue_module = module

                break


    if (
        continue_module is None
        and learning_modules
    ):

        continue_module = (
            learning_modules[0]
        )


    cursor.close()

    conn.close()

    return render_template(
        "index.html",

        # -------------------------------------------------
        # DETECTION COUNTS
        # -------------------------------------------------

        total_scans=total_scans,

        phishing_count=phishing_count,

        suspicious_count=suspicious_count,

        legitimate_count=legitimate_count,



        high_risk_count=high_risk_count,

        medium_risk_count=medium_risk_count,

        low_risk_count=low_risk_count,


        scam_reports_count=scam_reports_count,

        avg_confidence=avg_confidence,

        recent_scans=recent_scans,

        total_modules=total_modules,

        modules_completed=modules_completed,

        total_lessons=total_lessons,

        lessons_completed=lessons_completed,

        learning_progress=learning_progress,

        average_quiz_score=average_quiz_score,

        learning_modules=learning_modules,

        continue_module=continue_module,
    )


@app.route("/scanner")
@user_required
def scanner():

    return redirect(
        url_for(
            "sms_scanner"
        )
    )

def get_recommended_learning_modules(
    input_type,
    prediction,
    submitted_content="",
    reasons=None
):

    if prediction not in (
        "Phishing",
        "Malicious",
        "Suspicious",
    ):
        return []


    submitted_content = (
        submitted_content
        or ""
    ).lower()


    if reasons is None:
        reasons = []


    reason_text = " ".join(
        str(reason)
        for reason in reasons
    ).lower()


    analysis_text = (
        submitted_content
        + " "
        + reason_text
    )

    recommended_titles = []


    if input_type == "Email":

        recommended_titles.append(
            "Email Phishing"
        )


    elif input_type == "SMS":

        recommended_titles.append(
            "Smishing"
        )


    elif input_type == "URL":

        recommended_titles.append(
            "Malicious URLs"
        )

    banking_terms = (
        "bank",
        "banking",
        "account",
        "payment",
        "transaction",
        "transfer",
        "money",
        "refund",
        "card",
        "credit card",
        "debit card",
        "pin",
        "otp",
        "one-time pin",
        "cash",
        "payment failed",
        "account suspended",
        "account blocked",
        "verify payment",
        "verify account",
        "fnb",
        "absa",
        "standard bank",
        "nedbank",
        "capitec",
        "discovery bank",
    )


    if any(
        term in analysis_text
        for term in banking_terms
    ):

        recommended_titles.append(
            "Banking and Payment Scams"
        )

    credential_terms = (
        "password",
        "username",
        "login",
        "log in",
        "sign in",
        "signin",
        "credential",
        "credentials",
        "otp",
        "one-time pin",
        "verification code",
        "security code",
        "reset password",
        "confirm password",
        "verify identity",
        "verify your details",
        "account verification",
    )


    if any(
        term in analysis_text
        for term in credential_terms
    ):

        recommended_titles.append(
            "Account and Credential Theft"
        )

    social_engineering_terms = (
        "urgent",
        "urgently",
        "immediately",
        "act now",
        "final warning",
        "last warning",
        "suspended",
        "blocked",
        "prize",
        "winner",
        "won",
        "reward",
        "limited time",
        "verify immediately",
        "click immediately",
        "respond immediately",
        "failure to",
        "within 24 hours",
        "security alert",
    )


    if any(
        term in analysis_text
        for term in social_engineering_terms
    ):

        recommended_titles.append(
            "Social Engineering"
        )

    unique_titles = []

    for title in recommended_titles:

        if title not in unique_titles:

            unique_titles.append(
                title
            )

    unique_titles = unique_titles[:3]


    if not unique_titles:

        return []

    placeholders = ", ".join(
        ["%s"] * len(unique_titles)
    )


    conn = get_db_connection()

    cursor = conn.cursor(
        dictionary=True
    )


    query = f"""
        SELECT
            module_id,
            title,
            category,
            description,
            icon
        FROM learning_modules
        WHERE title IN ({placeholders})
    """


    cursor.execute(
        query,
        tuple(unique_titles)
    )


    modules = cursor.fetchall()


    cursor.close()

    conn.close()

    module_lookup = {
        module["title"]: module
        for module in modules
    }


    ordered_modules = []


    for title in unique_titles:

        module = module_lookup.get(
            title
        )

        if module:

            ordered_modules.append(
                module
            )


    return ordered_modules

@app.route("/sms-scanner", methods=["GET", "POST"])
@user_required
def sms_scanner():

    result = None

    if request.method == "POST":

        submitted_content = request.form["submitted_content"]

        prediction_result = predict_message(
            submitted_content
        )

        recommended_modules = get_recommended_learning_modules(
            input_type="SMS",
            prediction=prediction_result["prediction"],
            submitted_content=submitted_content,
            reasons=prediction_result.get(
                "reasons",
                []
            )
        )

        recommended_module = (
            recommended_modules[0]
            if recommended_modules
            else None
        )

        result = {

            "input_type":
                "SMS",

            "submitted_content":
                submitted_content,

            "prediction":
                prediction_result["prediction"],

            "confidence":
                round(
                    float(
                        prediction_result["confidence"]
                    ),
                    2
                ),

            "risk_level":
                prediction_result["risk_level"],

            "reasons":
                prediction_result.get(
                    "reasons",
                    []
                ),

            "actions":
                prediction_result.get(
                    "recommended_actions",
                    []
                ),

            "recommended_module":
                recommended_module,

            "recommended_modules":
                recommended_modules
        }

        save_detection(
            input_type="SMS",
            submitted_content=submitted_content,
            prediction_result=prediction_result,
        )

    return render_template(
        "sms_scanner.html",
        result=result
    )


@app.route("/email-scanner", methods=["GET", "POST"])
@user_required
def email_scanner():

    result = None

    if request.method == "POST":

        submitted_content = request.form["submitted_content"]

        # Run Email machine-learning prediction
        prediction_result = predict_email(
            submitted_content
        )

        # Get personalised learning recommendations
        recommended_modules = get_recommended_learning_modules(
            input_type="Email",
            prediction=prediction_result["prediction"],
            submitted_content=submitted_content,
            reasons=prediction_result.get(
                "reasons",
                []
            )
        )

        recommended_module = (
            recommended_modules[0]
            if recommended_modules
            else None
        )

        # Build result for the HTML page
        result = {

            "input_type":
                "Email",

            "submitted_content":
                submitted_content,

            "prediction":
                prediction_result["prediction"],

            "confidence":
                round(
                    float(
                        prediction_result["confidence"]
                    ),
                    2
                ),

            "risk_level":
                prediction_result["risk_level"],

            "reasons":
                prediction_result.get(
                    "reasons",
                    []
                ),

            "actions":
                prediction_result.get(
                    "recommended_actions",
                    []
                ),

            "recommended_module":
                recommended_module,

            "recommended_modules":
                recommended_modules
        }

        save_detection(
            input_type="Email",
            submitted_content=submitted_content,
            prediction_result=prediction_result,
        )

    return render_template(
        "email_scanner.html",
        result=result
    )


    return render_template(
        "email_scanner.html",
        result=result
    )


@app.route("/url-scanner", methods=["GET", "POST"])
@user_required
def url_scanner():

    result = None

    if request.method == "POST":

        submitted_content = request.form["submitted_content"]

        # Run URL machine-learning prediction
        prediction_result = predict_url(
            submitted_content
        )

        # Get personalised learning recommendations
        recommended_modules = get_recommended_learning_modules(
            input_type="URL",
            prediction=prediction_result["prediction"],
            submitted_content=submitted_content,
            reasons=prediction_result.get(
                "reasons",
                []
            )
        )

        recommended_module = (
            recommended_modules[0]
            if recommended_modules
            else None
        )

        # -------------------------------------------------
        # INVALID URL
        # -------------------------------------------------

        if prediction_result["prediction"] == "Invalid URL":

            result = {

                "input_type":
                    "URL",

                "submitted_content":
                    submitted_content,

                "prediction":
                    "Invalid URL",

                "validation_error":
                    prediction_result.get(
                        "validation_error",
                        "Please enter a valid website URL."
                    ),

                "reasons":
                    prediction_result.get(
                        "reasons",
                        []
                    ),

                "recommended_module":
                    None,

                "recommended_modules":
                    []
            }

        # -------------------------------------------------
        # VALID URL
        # -------------------------------------------------

        else:

            prediction = (
                prediction_result["prediction"]
            )

            confidence = round(
                float(
                    prediction_result["confidence"]
                ),
                2
            )

            legitimate_probability = round(
                float(
                    prediction_result.get(
                        "legitimate_probability",
                        0
                    )
                ),
                2
            )

            malicious_probability = round(
                float(
                    prediction_result.get(
                        "malicious_probability",
                        0
                    )
                ),
                2
            )

            model_display_prediction = (
                prediction
            )

            model_confidence = (
                confidence
            )

            decision_source = (
                "Machine-Learning Model"
            )

            actions = prediction_result.get(
                "recommended_actions",
                []
            )

            if not actions:

                if prediction == "Malicious":

                    actions = [
                        "Do not enter passwords, banking details or personal information.",
                        "Avoid downloading files from the website.",
                        "Do not continue using the link unless you can verify it.",
                        "Report the URL if you believe it is fraudulent or harmful."
                    ]

                else:

                    actions = [
                        "Continue to exercise normal online security precautions.",
                        "Check the website address before entering sensitive information.",
                        "Avoid downloading unexpected files."
                    ]

            result = {

                "input_type":
                    "URL",

                "submitted_content":
                    submitted_content,

                "prediction":
                    prediction,

                "confidence":
                    confidence,

                "risk_level":
                    prediction_result["risk_level"],

                "model_display_prediction":
                    model_display_prediction,

                "model_confidence":
                    model_confidence,

                "decision_source":
                    decision_source,

                "legitimate_probability":
                    legitimate_probability,

                "malicious_probability":
                    malicious_probability,

                "reasons":
                    prediction_result.get(
                        "reasons",
                        []
                    ),

                "actions":
                    actions,

                "warnings":
                    prediction_result.get(
                        "warnings",
                        []
                    ),

                "normalised_url":
                    prediction_result.get(
                        "normalised_url"
                    ),

                "url_segments":
                    prediction_result.get(
                        "url_segments",
                        []
                    ),

                "recommended_module":
                    recommended_module,

                "recommended_modules":
                    recommended_modules
            }


            save_detection(
                input_type="URL",
                submitted_content=submitted_content,
                prediction_result=prediction_result,
            )

    return render_template(
        "url_scanner.html",
        result=result
    )


@app.route("/history")
@user_required
def history():

    conn = get_db_connection()

    cursor = conn.cursor(
        dictionary=True
    )

    user_id = session["user_id"]


    cursor.execute(
        """
        SELECT
            detection_id,
            input_type,
            submitted_content,
            prediction,
            confidence_score,
            risk_level,
            explanation,
            scan_date
        FROM detections
        WHERE user_id = %s
        ORDER BY scan_date DESC
        """,
        (
            user_id,
        )
    )

    detections = (
        cursor.fetchall()
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
        """,
        (
            user_id,
        )
    )

    total_history = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
          AND input_type = 'Email'
        """,
        (
            user_id,
        )
    )

    email_count = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
          AND input_type = 'SMS'
        """,
        (
            user_id,
        )
    )

    sms_count = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM detections
        WHERE user_id = %s
          AND input_type = 'URL'
        """,
        (
            user_id,
        )
    )

    url_count = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.close()

    conn.close()


    return render_template(
        "history.html",

        detections=detections,

        total_history=total_history,

        email_count=email_count,

        sms_count=sms_count,

        url_count=url_count,
    )


@app.route(
    "/report-scam",
    methods=[
        "GET",
        "POST"
    ]
)
@user_required
def report_scam():

    success = False

    user_id = session["user_id"]


    report_type = request.form.get(
        "report_type",
        request.args.get(
            "report_type",
            "SMS"
        )
    )


    reported_content = request.form.get(
        "reported_content",
        request.args.get(
            "reported_content",
            ""
        )
    )


    description = request.form.get(
        "description",
        ""
    )


    

    allowed_report_types = {
        "Email",
        "SMS",
        "URL"
    }


    if report_type not in allowed_report_types:
        report_type = "SMS"


    if (
        request.method == "POST"
        and request.form.get(
            "submit_report"
        ) == "true"
    ):

        reported_content = (
            reported_content.strip()
        )

        description = (
            description.strip()
        )


        if not reported_content:

            flash(
                "Please enter the suspicious content you want to report.",
                "error"
            )

        else:

            conn = get_db_connection()

            cursor = conn.cursor()


            cursor.execute(
                """
                INSERT INTO scam_reports
                (
                    user_id,
                    report_type,
                    reported_content,
                    description
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    user_id,
                    report_type,
                    reported_content,
                    description,
                )
            )


            conn.commit()

            cursor.close()

            conn.close()


            success = True

            report_type = "SMS"

            reported_content = ""

            description = ""


    conn = get_db_connection()

    cursor = conn.cursor(
        dictionary=True
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE user_id = %s
        """,
        (
            user_id,
        )
    )

    total_reports = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE user_id = %s
          AND status = 'Pending'
        """,
        (
            user_id,
        )
    )

    pending_reports = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE user_id = %s
          AND status = 'Confirmed Scam'
        """,
        (
            user_id,
        )
    )

    confirmed_reports = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT
            report_id,
            report_type,
            reported_content,
            status,
            report_date
        FROM scam_reports
        WHERE user_id = %s
        ORDER BY report_date DESC
        LIMIT 3
        """,
        (
            user_id,
        )
    )

    recent_reports = (
        cursor.fetchall()
    )


    cursor.close()

    conn.close()


    return render_template(
        "report_scam.html",

        success=success,

        report_type=report_type,

        reported_content=reported_content,

        description=description,

        total_reports=total_reports,

        pending_reports=pending_reports,

        confirmed_reports=confirmed_reports,

        recent_reports=recent_reports,
    )

@app.route("/awareness")
@user_required
def awareness():

    user_id = session["user_id"]

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            lm.module_id,
            lm.title,
            lm.category,
            lm.description,
            lm.icon,
            lm.display_order,
            COALESCE(
                ulp.best_quiz_score,
                0
            ) AS best_quiz_score,
            COALESCE(
                ulp.module_completed,
                0
            ) AS module_completed
        FROM learning_modules lm
        LEFT JOIN user_learning_progress ulp
            ON ulp.module_id = lm.module_id
            AND ulp.user_id = %s
        ORDER BY lm.display_order
    """, (
        user_id,
    ))

    modules = cursor.fetchall()

    cursor.close()
    conn.close()

    for module in modules:

        progress = get_module_progress(
            user_id,
            module["module_id"],
        )

        module["lesson_count"] = (
            progress["total_lessons"]
        )

        module["lessons_completed"] = (
            progress["completed_lessons"]
        )

        module["progress_percentage"] = (
            progress["progress_percentage"]
        )

    scored_modules = [
        float(module["best_quiz_score"])
        for module in modules
        if float(module["best_quiz_score"]) > 0
    ]

    awareness_score = (
        round(
            sum(scored_modules)
            / len(scored_modules),
            2
        )
        if scored_modules
        else 0
    )

    return render_template(
        "awareness.html",
        modules=modules,
        awareness_score=awareness_score,
    )

@app.route("/scam-reports")
@user_required
def scam_reports():

    conn = get_db_connection()

    cursor = conn.cursor(
        dictionary=True
    )

    user_id = session["user_id"]

    cursor.execute(
        """
        SELECT
            report_id,
            report_type,
            reported_content,
            description,
            status,
            report_date
        FROM scam_reports
        WHERE user_id = %s
        ORDER BY report_date DESC
        """,
        (
            user_id,
        )
    )

    reports = cursor.fetchall()

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE user_id = %s
        """,
        (
            user_id,
        )
    )

    total_reports = (
        cursor.fetchone()["total"]
        or 0
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE user_id = %s
          AND status = 'Pending'
        """,
        (
            user_id,
        )
    )

    pending_reports = (
        cursor.fetchone()["total"]
        or 0
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE user_id = %s
          AND status = 'Reviewed'
        """,
        (
            user_id,
        )
    )

    reviewed_reports = (
        cursor.fetchone()["total"]
        or 0
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE user_id = %s
          AND status = 'Confirmed Scam'
        """,
        (
            user_id,
        )
    )

    confirmed_reports = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.close()

    conn.close()


    return render_template(
        "scam_reports.html",

        reports=reports,

        total_reports=total_reports,

        pending_reports=pending_reports,

        reviewed_reports=reviewed_reports,

        confirmed_reports=confirmed_reports,
    )


# =========================================================
# ADMIN - VIEW ALL SCAM REPORTS
# =========================================================

@app.route("/admin/scam-reports")
@admin_required
def admin_scam_reports():

    conn = get_db_connection()

    cursor = conn.cursor(
        dictionary=True
    )


    cursor.execute(
        """
        SELECT
            sr.report_id,
            sr.user_id,
            sr.report_type,
            sr.reported_content,
            sr.description,
            sr.status,
            sr.report_date,

            u.first_name,
            u.last_name,
            u.email

        FROM scam_reports sr

        INNER JOIN users u
            ON u.user_id = sr.user_id

        ORDER BY sr.report_date DESC
        """
    )

    reports = cursor.fetchall()


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        """
    )

    total_reports = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE status = 'Pending'
        """
    )

    pending_reports = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE status = 'Reviewed'
        """
    )

    reviewed_reports = (
        cursor.fetchone()["total"]
        or 0
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE status = 'Confirmed Scam'
        """
    )

    confirmed_reports = (
        cursor.fetchone()["total"]
        or 0
    )

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM scam_reports
        WHERE status = 'Not a Scam'
        """
    )

    not_scam_reports = (
        cursor.fetchone()["total"]
        or 0
    )


    cursor.close()
    conn.close()


    return render_template(
        "admin_side_scam_reports.html",

        reports=reports,

        total_reports=total_reports,

        pending_reports=pending_reports,

        reviewed_reports=reviewed_reports,

        confirmed_reports=confirmed_reports,

        not_scam_reports=not_scam_reports,
    )


@app.route(
    "/admin/scam-reports/<int:report_id>/status",
    methods=["POST"]
)
@admin_required
def admin_update_report_status(
    report_id
):

    new_status = request.form.get(
        "status",
        ""
    )


    allowed_statuses = {
        "Pending",
        "Reviewed",
        "Confirmed Scam",
        "Not a Scam",
    }


    if new_status not in allowed_statuses:

        flash(
            "Invalid report status.",
            "error"
        )

        return redirect(
            url_for(
                "admin_scam_reports"
            )
        )


    conn = get_db_connection()

    cursor = conn.cursor()


    cursor.execute(
        """
        UPDATE scam_reports
        SET status = %s
        WHERE report_id = %s
        """,
        (
            new_status,
            report_id,
        )
    )


    conn.commit()

    cursor.close()

    conn.close()


    flash(
        "Report status updated successfully.",
        "success"
    )


    return redirect(
        url_for(
            "admin_scam_reports"
        )
    )

@app.route(
    "/register",
    methods=[
        "GET",
        "POST"
    ]
)
def register():

    if request.method == "POST":

        first_name = (
            request.form[
                "first_name"
            ]
            .strip()
        )

        last_name = (
            request.form[
                "last_name"
            ]
            .strip()
        )

        email = (
            request.form[
                "email"
            ]
            .strip()
            .lower()
        )

        password = (
            request.form[
                "password"
            ]
        )

        confirm_password = (
            request.form[
                "confirm_password"
            ]
        )

        if (
            password
            != confirm_password
        ):

            flash(
                "The passwords do not match.",
                "error"
            )

            return render_template(
                "register.html"
            )

        if len(password) < 8:

            flash(
                "The password must contain at least 8 characters.",
                "error"
            )

            return render_template(
                "register.html"
            )

        connection = (
            get_db_connection()
        )

        cursor = (
            connection.cursor(
                dictionary=True
            )
        )

        cursor.execute(
            """
            SELECT user_id
            FROM users
            WHERE email = %s
            """,
            (
                email,
            )
        )

        existing_user = (
            cursor.fetchone()
        )

        if existing_user:

            cursor.close()
            connection.close()

            flash(
                "An account with this email already exists.",
                "error"
            )

            return render_template(
                "register.html"
            )

        password_hash = (
            generate_password_hash(
                password
            )
        )

        cursor.execute(
            """
            INSERT INTO users
            (
                first_name,
                last_name,
                email,
                password_hash
            )
            VALUES (%s, %s, %s, %s)
            """,
            (
                first_name,
                last_name,
                email,
                password_hash,
            )
        )

        connection.commit()
        cursor.close()
        connection.close()

        flash(
            "Registration successful. You can now log in.",
            "success"
        )

        return redirect(
            url_for(
                "login"
            )
        )

    return render_template(
        "register.html"
    )


@app.route(
    "/login",
    methods=[
        "GET",
        "POST"
    ]
)
def login():

    if request.method == "POST":

        email = (
            request.form[
                "email"
            ]
            .strip()
            .lower()
        )

        password = (
            request.form[
                "password"
            ]
        )


        connection = (
            get_db_connection()
        )

        cursor = (
            connection.cursor(
                dictionary=True
            )
        )


        cursor.execute(
            """
            SELECT
                user_id,
                first_name,
                last_name,
                email,
                password_hash,
                role
            FROM users
            WHERE email = %s
            """,
            (
                email,
            )
        )


        user = (
            cursor.fetchone()
        )


        cursor.close()
        connection.close()


        if (
            user
            and check_password_hash(
                user[
                    "password_hash"
                ],
                password
            )
        ):

            session.clear()


            session[
                "user_id"
            ] = user[
                "user_id"
            ]


            session[
                "first_name"
            ] = user[
                "first_name"
            ]


            session[
                "role"
            ] = user[
                "role"
            ]
            if (
                str(
                    user["role"]
                )
                .strip()
                .lower()
                == "admin"
            ):

                return redirect(
                    url_for(
                        "admin_scam_reports"
                    )
                )

            return redirect(
                url_for(
                    "home"
                )
            )


        flash(
            "Invalid email address or password.",
            "error"
        )


    return render_template(
        "login.html"
    )


@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(
        url_for(
            "login"
        )
    )

@app.route("/learning/module/<int:module_id>")
@user_required
def learning_module(module_id):

    user_id = session["user_id"]

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM learning_modules
        WHERE module_id = %s
    """, (
        module_id,
    ))

    module = cursor.fetchone()

    if not module:
        cursor.close()
        conn.close()
        return "Learning module not found.", 404

    cursor.execute("""
        SELECT
            ll.lesson_id,
            ll.module_id,
            ll.title,
            ll.lesson_content,
            ll.display_order,
            COALESCE(
                ulp.completed,
                0
            ) AS completed
        FROM learning_lessons ll
        LEFT JOIN user_lesson_progress ulp
            ON ulp.lesson_id = ll.lesson_id
            AND ulp.user_id = %s
        WHERE ll.module_id = %s
        ORDER BY ll.display_order
    """, (
        user_id,
        module_id,
    ))

    lessons = cursor.fetchall()

    cursor.close()
    conn.close()

    total_lessons = len(lessons)

    completed_lessons = sum(
        1
        for lesson in lessons
        if lesson["completed"]
    )

    all_lessons_completed = (
        total_lessons > 0
        and completed_lessons == total_lessons
    )

    progress_percentage = (
        round(
            (
                completed_lessons
                / total_lessons
            ) * 100,
            2
        )
        if total_lessons > 0
        else 0
    )

    return render_template(
        "learning_module.html",
        module=module,
        lessons=lessons,
        total_lessons=total_lessons,
        completed_lessons=completed_lessons,
        all_lessons_completed=all_lessons_completed,
        progress_percentage=progress_percentage,
    )


@app.route(
    "/learning/module/<int:module_id>/lesson/<int:lesson_id>/complete",
    methods=["POST"]
)
@user_required
def complete_learning_lesson(module_id, lesson_id):

    user_id = session["user_id"]

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT lesson_id
        FROM learning_lessons
        WHERE lesson_id = %s
          AND module_id = %s
    """, (
        lesson_id,
        module_id,
    ))

    lesson = cursor.fetchone()

    if not lesson:
        cursor.close()
        conn.close()

        return {
            "success": False,
            "message": "Lesson not found."
        }, 404

    cursor.close()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO user_lesson_progress
        (
            user_id,
            lesson_id,
            completed,
            completed_at
        )
        VALUES
        (
            %s,
            %s,
            1,
            NOW()
        )
        ON DUPLICATE KEY UPDATE
            completed = 1,
            completed_at = NOW()
    """, (
        user_id,
        lesson_id,
    ))

    conn.commit()
    cursor.close()
    conn.close()

    progress = get_module_progress(
        user_id,
        module_id,
    )

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO user_learning_progress
        (
            user_id,
            module_id,
            lessons_completed,
            module_completed,
            best_quiz_score
        )
        VALUES
        (
            %s,
            %s,
            %s,
            0,
            0
        )
        ON DUPLICATE KEY UPDATE
            lessons_completed = %s
    """, (
        user_id,
        module_id,
        progress["completed_lessons"],
        progress["completed_lessons"],
    ))

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "success": True,
        "completed_lessons":
            progress["completed_lessons"],
        "total_lessons":
            progress["total_lessons"],
        "progress_percentage":
            progress["progress_percentage"],
        "all_lessons_completed":
            progress["all_lessons_completed"],
    }


@app.route(
    "/learning/module/<int:module_id>/restart",
    methods=["POST"]
)
@user_required
def restart_learning_module(module_id):

    user_id = session["user_id"]

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        DELETE ulp
        FROM user_lesson_progress ulp
        INNER JOIN learning_lessons ll
            ON ll.lesson_id = ulp.lesson_id
        WHERE ulp.user_id = %s
          AND ll.module_id = %s
    """, (
        user_id,
        module_id,
    ))

    cursor.execute("""
        DELETE FROM user_learning_progress
        WHERE user_id = %s
          AND module_id = %s
    """, (
        user_id,
        module_id,
    ))

    conn.commit()
    cursor.close()
    conn.close()

    flash(
        "Module progress has been restarted.",
        "success"
    )

    return redirect(
        url_for(
            "learning_module",
            module_id=module_id,
        )
    )


@app.route("/learning/module/<int:module_id>/quiz")
@user_required
def module_quiz(module_id):

    user_id = session["user_id"]

    progress = get_module_progress(
        user_id,
        module_id,
    )

    if not progress["all_lessons_completed"]:

        flash(
            "Complete all lessons before taking the quiz.",
            "warning"
        )

        return redirect(
            url_for(
                "learning_module",
                module_id=module_id,
            )
        )

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM learning_modules
        WHERE module_id = %s
    """, (
        module_id,
    ))

    module = cursor.fetchone()

    if not module:
        cursor.close()
        conn.close()
        return "Learning module not found.", 404

    cursor.execute("""
        SELECT
            question_id,
            question_text,
            option_a,
            option_b,
            option_c,
            option_d,
            display_order
        FROM quiz_questions
        WHERE module_id = %s
        ORDER BY display_order
    """, (
        module_id,
    ))

    questions = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template(
        "module_quiz.html",
        module=module,
        questions=questions,
    )


@app.route(
    "/learning/module/<int:module_id>/quiz/submit",
    methods=["POST"]
)
@user_required
def submit_module_quiz(module_id):

    user_id = session["user_id"]

    progress = get_module_progress(
        user_id,
        module_id,
    )

    if not progress["all_lessons_completed"]:

        flash(
            "Complete all lessons before submitting the quiz.",
            "warning"
        )

        return redirect(
            url_for(
                "learning_module",
                module_id=module_id,
            )
        )

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT *
        FROM learning_modules
        WHERE module_id = %s
    """, (
        module_id,
    ))

    module = cursor.fetchone()

    if not module:
        cursor.close()
        conn.close()
        return "Learning module not found.", 404

    cursor.execute("""
        SELECT
            question_id,
            question_text,
            option_a,
            option_b,
            option_c,
            option_d,
            correct_option,
            explanation
        FROM quiz_questions
        WHERE module_id = %s
        ORDER BY display_order
    """, (
        module_id,
    ))

    questions = cursor.fetchall()

    if not questions:
        cursor.close()
        conn.close()

        flash(
            "This module does not have quiz questions yet.",
            "warning"
        )

        return redirect(
            url_for(
                "learning_module",
                module_id=module_id,
            )
        )

    score = 0
    results = []

    for question in questions:

        selected_option = request.form.get(
            f"question_{question['question_id']}"
        )

        correct_option = (
            question["correct_option"]
        )

        is_correct = (
            selected_option == correct_option
        )

        if is_correct:
            score += 1

        option_lookup = {
            "A": question["option_a"],
            "B": question["option_b"],
            "C": question["option_c"],
            "D": question["option_d"],
        }

        results.append({
            "question_text":
                question["question_text"],
            "selected_option":
                selected_option,
            "selected_answer":
                option_lookup.get(
                    selected_option,
                    "No answer"
                ),
            "correct_option":
                correct_option,
            "correct_answer":
                option_lookup[
                    correct_option
                ],
            "is_correct":
                is_correct,
            "explanation":
                question["explanation"],
        })

    total_questions = len(
        questions
    )

    percentage = round(
        (
            score
            / total_questions
        ) * 100,
        2
    )

    passed = (
        percentage >= 70
    )

    cursor.close()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO quiz_attempts
        (
            user_id,
            module_id,
            score,
            total_questions,
            percentage
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s
        )
    """, (
        user_id,
        module_id,
        score,
        total_questions,
        percentage,
    ))

    cursor.execute("""
        INSERT INTO user_learning_progress
        (
            user_id,
            module_id,
            lessons_completed,
            module_completed,
            best_quiz_score
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s
        )
        ON DUPLICATE KEY UPDATE
            lessons_completed = %s,
            module_completed =
                GREATEST(
                    module_completed,
                    %s
                ),
            best_quiz_score =
                GREATEST(
                    best_quiz_score,
                    %s
                )
    """, (
        user_id,
        module_id,
        progress["completed_lessons"],
        int(passed),
        percentage,
        progress["completed_lessons"],
        int(passed),
        percentage,
    ))

    conn.commit()
    cursor.close()
    conn.close()

    return render_template(
        "quiz_result.html",
        module=module,
        score=score,
        total_questions=total_questions,
        percentage=percentage,
        passed=passed,
        results=results,
    )


if __name__ == "__main__":
    app.run(
        debug=True
    )