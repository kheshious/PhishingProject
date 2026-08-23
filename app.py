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
@login_required
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

    cursor.close()
    conn.close()

    return render_template(
        "index.html",
        total_scans=total_scans,
        phishing_count=phishing_count,
        suspicious_count=suspicious_count,
        legitimate_count=legitimate_count,
        scam_reports_count=scam_reports_count,
        avg_confidence=avg_confidence,
        recent_scans=recent_scans,
    )


@app.route("/scanner")
@login_required
def scanner():

    return redirect(
        url_for(
            "sms_scanner"
        )
    )


@app.route(
    "/sms-scanner",
    methods=[
        "GET",
        "POST"
    ]
)
@login_required
def sms_scanner():

    result = None

    if request.method == "POST":

        submitted_content = (
            request.form.get(
                "submitted_content",
                ""
            ).strip()
        )

        if not submitted_content:

            flash(
                "Please enter an SMS message.",
                "warning"
            )

            return render_template(
                "sms_scanner.html",
                result=None
            )

        prediction_result = (
            predict_message(
                submitted_content
            )
        )

        result = {
            "input_type": "SMS",
            "submitted_content":
                submitted_content,
            "prediction":
                prediction_result[
                    "prediction"
                ],
            "confidence": round(
                prediction_result[
                    "confidence"
                ],
                2
            ),
            "risk_level":
                prediction_result[
                    "risk_level"
                ],
            "reasons":
                prediction_result[
                    "reasons"
                ],
            "actions":
                prediction_result[
                    "recommended_actions"
                ],
            "legitimate_probability":
                prediction_result.get(
                    "legitimate_probability"
                ),
            "smishing_probability":
                prediction_result.get(
                    "smishing_probability"
                ),
        }

        save_detection(
            "SMS",
            submitted_content,
            prediction_result,
        )

    return render_template(
        "sms_scanner.html",
        result=result
    )


@app.route(
    "/email-scanner",
    methods=[
        "GET",
        "POST"
    ]
)
@login_required
def email_scanner():

    result = None

    if request.method == "POST":

        submitted_content = (
            request.form.get(
                "submitted_content",
                ""
            ).strip()
        )

        if not submitted_content:

            flash(
                "Please enter email content.",
                "warning"
            )

            return render_template(
                "email_scanner.html",
                result=None
            )

        prediction_result = (
            predict_email(
                submitted_content
            )
        )

        result = {
            "input_type": "Email",
            "submitted_content":
                submitted_content,
            "prediction":
                prediction_result[
                    "prediction"
                ],
            "confidence": round(
                prediction_result[
                    "confidence"
                ],
                2
            ),
            "risk_level":
                prediction_result[
                    "risk_level"
                ],
            "reasons":
                prediction_result[
                    "reasons"
                ],
            "actions":
                prediction_result[
                    "recommended_actions"
                ],
            "legitimate_probability":
                prediction_result.get(
                    "legitimate_probability"
                ),
            "phishing_probability":
                prediction_result.get(
                    "phishing_probability"
                ),
        }

        save_detection(
            "Email",
            submitted_content,
            prediction_result,
        )

    return render_template(
        "email_scanner.html",
        result=result
    )


@app.route(
    "/url-scanner",
    methods=[
        "GET",
        "POST"
    ]
)
@login_required
def url_scanner():

    result = None

    if request.method == "POST":

        submitted_content = (
            request.form.get(
                "submitted_content",
                ""
            ).strip()
        )

        if not submitted_content:

            flash(
                "Please enter a website URL.",
                "warning"
            )

            return render_template(
                "url_scanner.html",
                result=None
            )

        prediction_result = (
            predict_url(
                submitted_content
            )
        )

        if not prediction_result[
            "is_valid"
        ]:

            result = {
                "input_type": "URL",
                "submitted_content":
                    submitted_content,
                "is_valid": False,
                "prediction":
                    "Invalid URL",
                "confidence": 0,
                "risk_level": "None",
                "validation_error":
                    prediction_result[
                        "validation_error"
                    ],
                "reasons":
                    prediction_result[
                        "reasons"
                    ],
                "actions":
                    prediction_result[
                        "recommended_actions"
                    ],
                "warnings": [],
                "url_segments": [],
            }

            return render_template(
                "url_scanner.html",
                result=result
            )

        result = {
            "input_type": "URL",
            "submitted_content":
                submitted_content,

            "is_valid": True,

            "prediction":
                prediction_result[
                    "prediction"
                ],

            "confidence": round(
                prediction_result[
                    "confidence"
                ],
                2
            ),

            "risk_level":
                prediction_result[
                    "risk_level"
                ],

            "decision_source":
                prediction_result[
                    "decision_source"
                ],

            "model_prediction":
                prediction_result[
                    "model_prediction"
                ],

            "model_display_prediction":
                prediction_result[
                    "model_display_prediction"
                ],

            "model_confidence":
                prediction_result[
                    "model_confidence"
                ],

            "legitimate_probability":
                prediction_result[
                    "legitimate_probability"
                ],

            "malicious_probability":
                prediction_result[
                    "malicious_probability"
                ],

            "class_probabilities":
                prediction_result[
                    "class_probabilities"
                ],

            "reasons":
                prediction_result[
                    "reasons"
                ],

            "warnings":
                prediction_result[
                    "warnings"
                ],

            "actions":
                prediction_result[
                    "recommended_actions"
                ],

            "url_segments":
                prediction_result[
                    "url_segments"
                ],

            "hostname":
                prediction_result[
                    "hostname"
                ],

            "normalised_url":
                prediction_result[
                    "normalised_url"
                ],
        }

        save_detection(
            "URL",
            submitted_content,
            prediction_result,
        )

    return render_template(
        "url_scanner.html",
        result=result
    )


@app.route("/history")
@login_required
def history():

    conn = get_db_connection()
    cursor = conn.cursor(
        dictionary=True
    )

    cursor.execute(
        """
        SELECT *
        FROM detections
        WHERE user_id = %s
        ORDER BY scan_date DESC
        """,
        (
            session["user_id"],
        )
    )

    detections = (
        cursor.fetchall()
    )

    cursor.close()
    conn.close()

    return render_template(
        "history.html",
        detections=detections
    )


@app.route(
    "/report-scam",
    methods=[
        "GET",
        "POST"
    ]
)
@login_required
def report_scam():

    success = False

    report_type = (
        request.form.get(
            "report_type",
            request.args.get(
                "report_type",
                "SMS"
            )
        )
    )

    reported_content = (
        request.form.get(
            "reported_content",
            request.args.get(
                "reported_content",
                ""
            )
        )
    )

    description = (
        request.form.get(
            "description",
            ""
        )
    )

    if (
        request.method == "POST"
        and request.form.get(
            "submit_report"
        ) == "true"
    ):

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
            VALUES (%s, %s, %s, %s)
            """,
            (
                session["user_id"],
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

    return render_template(
        "report_scam.html",
        success=success,
        report_type=report_type,
        reported_content=reported_content,
        description=description,
    )


@app.route("/awareness")
@login_required
def awareness():

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

            COUNT(DISTINCT ll.lesson_id) AS lesson_count,

            COALESCE(
                ulp.lessons_completed,
                0
            ) AS lessons_completed,

            COALESCE(
                ulp.best_quiz_score,
                0
            ) AS best_quiz_score,

            COALESCE(
                ulp.module_completed,
                0
            ) AS module_completed

        FROM learning_modules lm

        LEFT JOIN learning_lessons ll
            ON ll.module_id = lm.module_id

        LEFT JOIN user_learning_progress ulp
            ON ulp.module_id = lm.module_id
            AND ulp.user_id = %s

        GROUP BY
            lm.module_id,
            lm.title,
            lm.category,
            lm.description,
            lm.icon,
            lm.display_order,
            ulp.lessons_completed,
            ulp.best_quiz_score,
            ulp.module_completed

        ORDER BY lm.display_order
    """, (
        session["user_id"],
    ))

    modules = cursor.fetchall()

    cursor.execute("""
        SELECT
            ROUND(
                AVG(best_quiz_score),
                2
            ) AS awareness_score
        FROM user_learning_progress
        WHERE user_id = %s
    """, (
        session["user_id"],
    ))

    awareness_score = (
        cursor.fetchone()["awareness_score"]
        or 0
    )

    cursor.close()
    conn.close()

    return render_template(
        "awareness.html",
        modules=modules,
        awareness_score=awareness_score,
    )


@app.route("/scam-reports")
@login_required
def scam_reports():

    conn = get_db_connection()
    cursor = conn.cursor(
        dictionary=True
    )

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
            session["user_id"],
        )
    )

    reports = (
        cursor.fetchall()
    )

    cursor.close()
    conn.close()

    return render_template(
        "scam_reports.html",
        reports=reports
    )


@app.route(
    "/scam-reports/<int:report_id>/status",
    methods=["POST"]
)
@login_required
def update_report_status(
    report_id
):

    new_status = (
        request.form.get(
            "status"
        )
    )

    allowed_statuses = {
        "Pending",
        "Reviewed",
        "Confirmed Scam",
        "Not a Scam",
    }

    if (
        new_status
        not in allowed_statuses
    ):
        return (
            "Invalid status",
            400
        )

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE scam_reports
        SET status = %s
        WHERE report_id = %s
        AND user_id = %s
        """,
        (
            new_status,
            report_id,
            session["user_id"],
        )
    )

    conn.commit()
    cursor.close()
    conn.close()

    return redirect(
        url_for(
            "scam_reports"
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
@login_required
def learning_module(module_id):

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
        session["user_id"],
        module_id,
    ))

    lessons = cursor.fetchall()

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

    cursor.close()
    conn.close()

    return render_template(
        "learning_module.html",
        module=module,
        lessons=lessons,
        total_lessons=total_lessons,
        completed_lessons=completed_lessons,
        all_lessons_completed=all_lessons_completed,
    )

@app.route("/learning/module/<int:module_id>/quiz")
@login_required
def module_quiz(module_id):

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
@login_required
def submit_module_quiz(module_id):

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

    score = 0
    results = []

    for question in questions:

        selected_option = request.form.get(
            f"question_{question['question_id']}"
        )

        correct_option = question["correct_option"]

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

    percentage = (
        (score / total_questions) * 100
        if total_questions > 0
        else 0
    )

    insert_cursor = conn.cursor()

    insert_cursor.execute("""
        INSERT INTO quiz_attempts
        (
            user_id,
            module_id,
            score,
            total_questions,
            percentage
        )
        VALUES (%s, %s, %s, %s, %s)
    """, (
        session["user_id"],
        module_id,
        score,
        total_questions,
        percentage,
    ))

    insert_cursor.execute("""
        INSERT INTO user_learning_progress
        (
            user_id,
            module_id,
            lessons_completed,
            module_completed,
            best_quiz_score
        )
        VALUES (%s, %s, %s, %s, %s)

        ON DUPLICATE KEY UPDATE

            lessons_completed =
                GREATEST(
                    lessons_completed,
                    VALUES(lessons_completed)
                ),

            module_completed =
                GREATEST(
                    module_completed,
                    VALUES(module_completed)
                ),

            best_quiz_score =
                GREATEST(
                    best_quiz_score,
                    VALUES(best_quiz_score)
                )
    """, (
        session["user_id"],
        module_id,
        5,
        percentage >= 70,
        percentage,
    ))

    conn.commit()

    insert_cursor.close()
    cursor.close()
    conn.close()

    return render_template(
        "quiz_result.html",
        module=module,
        score=score,
        total_questions=total_questions,
        percentage=round(
            percentage,
            2
        ),
        results=results,
    )

@app.route(
    "/learning/module/<int:module_id>/lesson/<int:lesson_id>/complete",
    methods=["POST"]
)
@login_required
def complete_learning_lesson(
    module_id,
    lesson_id
):

    user_id = session["user_id"]

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)


    # Make sure this lesson actually belongs
    # to the requested module
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


    # Save lesson completion.
    # If it already exists, simply keep it completed.
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
            completed_at = COALESCE(
                completed_at,
                NOW()
            )
    """, (
        user_id,
        lesson_id,
    ))

    conn.commit()


    # Calculate current module progress
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


    progress_percentage = 0

    if total_lessons > 0:

        progress_percentage = round(
            (
                completed_lessons
                / total_lessons
            ) * 100
        )


    return {
        "success": True,
        "completed_lessons": completed_lessons,
        "total_lessons": total_lessons,
        "progress_percentage": progress_percentage,
        "all_lessons_completed": (
            total_lessons > 0
            and
            completed_lessons == total_lessons
        )
    }

if __name__ == "__main__":
    app.run(
        debug=True
    )