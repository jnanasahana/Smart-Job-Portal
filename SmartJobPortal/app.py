import os
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, flash, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import functools
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smart_jobs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# ==================== MODELS ==================== #

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    user_type = db.Column(db.String(20), default='job_seeker')  # job_seeker, employer, admin
    phone = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Profile fields
    headline = db.Column(db.String(200))
    bio = db.Column(db.Text)
    skills = db.Column(db.Text)  # comma separated
    experience = db.Column(db.Text)
    education = db.Column(db.Text)
    location = db.Column(db.String(100))
    resume_text = db.Column(db.Text)
    
    # Preferences
    preferred_locations = db.Column(db.Text)
    salary_expectation = db.Column(db.Integer)
    remote_preference = db.Column(db.Boolean, default=False)
    
    # Relationships
    jobs_posted = db.relationship('Job', backref='employer', lazy=True)
    applications = db.relationship('Application', backref='applicant', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_full_name(self):
        return f"{self.first_name} {self.last_name}"
    
    def get_skills_list(self):
        if self.skills:
            return [skill.strip() for skill in self.skills.split(',')]
        return []
    
    def is_job_seeker(self):
        return self.user_type == 'job_seeker'
    
    def is_employer(self):
        return self.user_type == 'employer'
    
    def __repr__(self):
        return f'<User {self.email}>'

class Job(db.Model):
    __tablename__ = 'jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False, index=True)
    description = db.Column(db.Text, nullable=False)
    requirements = db.Column(db.Text)
    
    # Company info
    company_name = db.Column(db.String(200), nullable=False, index=True)
    company_website = db.Column(db.String(200))
    company_description = db.Column(db.Text)
    
    # Job details
    job_type = db.Column(db.String(50))  # full-time, part-time, contract, internship, remote
    experience_level = db.Column(db.String(50))  # entry, mid, senior, executive
    education_required = db.Column(db.String(100))
    
    # Location
    location = db.Column(db.String(200), index=True)
    city = db.Column(db.String(100))
    country = db.Column(db.String(100))
    is_remote = db.Column(db.Boolean, default=False)
    is_hybrid = db.Column(db.Boolean, default=False)
    
    # Salary
    salary_min = db.Column(db.Integer)
    salary_max = db.Column(db.Integer)
    salary_currency = db.Column(db.String(10), default='USD')
    
    # Status
    status = db.Column(db.String(20), default='active')  # active, closed, draft
    is_featured = db.Column(db.Boolean, default=False)
    views_count = db.Column(db.Integer, default=0)
    
    # Relationships
    posted_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Timestamps
    posted_date = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    expiry_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    applications = db.relationship('Application', backref='job', lazy=True)
    
    def get_formatted_salary(self):
        if self.salary_min and self.salary_max:
            return f"${self.salary_min:,} - ${self.salary_max:,} {self.salary_currency}"
        elif self.salary_min:
            return f"From ${self.salary_min:,} {self.salary_currency}"
        return "Not specified"
    
    def increment_views(self):
        self.views_count += 1
        db.session.commit()
    
    def __repr__(self):
        return f'<Job {self.title}>'

class Application(db.Model):
    __tablename__ = 'applications'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'), nullable=False, index=True)
    applicant_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    
    # Application details
    cover_letter = db.Column(db.Text)
    resume_text = db.Column(db.Text)
    
    # Status
    status = db.Column(db.String(20), default='pending')  # pending, reviewed, shortlisted, rejected, hired
    ai_score = db.Column(db.Float, default=0.0)
    ai_feedback = db.Column(db.Text)
    
    # Timestamps
    applied_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Application {self.id}>'

# ==================== AI JOB MATCHER ==================== #

class JobMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        # Skill categories
        self.skill_categories = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust', 'php', 'swift'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'express'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle'],
            'cloud': ['aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'terraform'],
            'data': ['machine learning', 'ai', 'data science', 'pandas', 'numpy', 'tensorflow', 'pytorch'],
            'mobile': ['android', 'ios', 'react native', 'flutter'],
            'devops': ['jenkins', 'git', 'ci/cd', 'ansible', 'prometheus']
        }
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_skills(self, text):
        """Extract skills from text"""
        text_lower = text.lower()
        found_skills = []
        
        for category, skills in self.skill_categories.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills.append(skill)
        
        return list(set(found_skills))
    
    def calculate_match_score(self, user_data, job_data):
        """Calculate match score between user and job"""
        # Extract user skills
        user_skills = set(user_data.get('skills', []))
        user_experience = user_data.get('experience', '')
        user_preferences = user_data.get('preferences', {})
        
        # Extract job data
        job_text = f"{job_data.get('title', '')} {job_data.get('description', '')}"
        job_skills = set(self.extract_skills(job_text))
        job_location = job_data.get('location', '').lower()
        
        # Skill match
        if user_skills and job_skills:
            skill_match = len(user_skills.intersection(job_skills)) / len(job_skills)
        else:
            skill_match = 0
        
        # Text similarity
        user_text = ' '.join(user_skills) + ' ' + user_experience
        job_text_clean = self.preprocess_text(job_text)
        user_text_clean = self.preprocess_text(user_text)
        
        if user_text_clean and job_text_clean:
            try:
                tfidf_matrix = self.vectorizer.fit_transform([user_text_clean, job_text_clean])
                text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                text_similarity = 0
        else:
            text_similarity = 0
        
        # Location match
        location_match = 0.5  # neutral
        preferred_locations = user_preferences.get('locations', [])
        
        if preferred_locations and job_location:
            for pref_loc in preferred_locations:
                if pref_loc.lower() in job_location or job_location in pref_loc.lower():
                    location_match = 1.0
                    break
            if 'remote' in job_location and user_preferences.get('remote', False):
                location_match = 0.8
        
        # Salary match
        salary_match = 0.5
        user_salary = user_preferences.get('salary', 0)
        job_min = job_data.get('salary_min', 0)
        job_max = job_data.get('salary_max', job_min * 2 if job_min else 0)
        
        if user_salary and job_min:
            if job_min <= user_salary <= job_max:
                salary_match = 1.0
            elif user_salary < job_min:
                salary_match = 0.7
            else:
                salary_match = 0.3
        
        # Overall score (weighted)
        overall_score = (
            skill_match * 0.3 +
            text_similarity * 0.3 +
            location_match * 0.2 +
            salary_match * 0.2
        )
        
        return {
            'overall_score': min(overall_score * 100, 100),  # Cap at 100%
            'skill_match': skill_match * 100,
            'text_similarity': text_similarity * 100,
            'location_match': location_match * 100,
            'salary_match': salary_match * 100,
            'matched_skills': list(user_skills.intersection(job_skills)),
            'missing_skills': list(job_skills - user_skills)
        }
    
    def get_recommendations(self, user, limit=10):
        """Get job recommendations for a user"""
        # Prepare user data
        user_data = {
            'skills': user.get_skills_list(),
            'experience': user.experience or '',
            'preferences': {
                'locations': user.preferred_locations.split(',') if user.preferred_locations else [],
                'salary': user.salary_expectation or 0,
                'remote': user.remote_preference
            }
        }
        
        # Get all active jobs
        jobs = Job.query.filter_by(status='active').all()
        
        recommendations = []
        for job in jobs:
            job_data = {
                'id': job.id,
                'title': job.title,
                'description': job.description,
                'location': job.location,
                'salary_min': job.salary_min,
                'salary_max': job.salary_max,
                'company_name': job.company_name,
                'job_type': job.job_type,
                'is_remote': job.is_remote
            }
            
            match_result = self.calculate_match_score(user_data, job_data)
            
            recommendations.append({
                'job': job,
                'match_scores': match_result
            })
        
        # Sort by overall score
        recommendations.sort(key=lambda x: x['match_scores']['overall_score'], reverse=True)
        
        return recommendations[:limit]

# Initialize job matcher
job_matcher = JobMatcher()

# ==================== DECORATORS ==================== #

def employer_required(f):
    """Decorator to restrict access to employers only"""
    @functools.wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_employer():
            flash('This page is for employers only.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# ==================== ROUTES ==================== #

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    """Home page"""
    featured_jobs = Job.query.filter_by(status='active', is_featured=True)\
        .order_by(Job.posted_date.desc()).limit(6).all()
    
    return render_template('index.html', 
                         featured_jobs=featured_jobs,
                         user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        user_type = request.form.get('user_type', 'job_seeker')
        
        # Check if user exists
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        user = User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            user_type=user_type
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """Logout user"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    if current_user.is_job_seeker():
        # Get user's applications
        applications = Application.query.filter_by(applicant_id=current_user.id)\
            .order_by(Application.applied_date.desc()).limit(10).all()
        
        # Get recommended jobs
        recommendations = job_matcher.get_recommendations(current_user, limit=5)
        
        return render_template('dashboard.html',
                             user=current_user,
                             applications=applications,
                             recommendations=recommendations,
                             is_seeker=True)
    
    else:  # Employer
        # Get employer's jobs
        jobs_posted = Job.query.filter_by(posted_by=current_user.id).all()
        
        # Get total applications
        total_applications = 0
        for job in jobs_posted:
            total_applications += len(job.applications)
        
        return render_template('dashboard.html',
                             user=current_user,
                             jobs_posted=jobs_posted,
                             total_applications=total_applications,
                             is_seeker=False)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile page"""
    if request.method == 'POST':
        # Update user info
        current_user.first_name = request.form.get('first_name')
        current_user.last_name = request.form.get('last_name')
        current_user.phone = request.form.get('phone')
        
        # Update profile info
        current_user.headline = request.form.get('headline')
        current_user.bio = request.form.get('bio')
        current_user.skills = request.form.get('skills')
        current_user.experience = request.form.get('experience')
        current_user.education = request.form.get('education')
        current_user.location = request.form.get('location')
        
        # Update preferences
        current_user.preferred_locations = request.form.get('preferred_locations')
        current_user.salary_expectation = request.form.get('salary_expectation')
        current_user.remote_preference = 'remote_preference' in request.form
        
        # Update resume text
        current_user.resume_text = request.form.get('resume_text')
        
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', user=current_user)

@app.route('/jobs')
def jobs():
    """Job listings page"""
    search = request.args.get('search', '')
    location = request.args.get('location', '')
    job_type = request.args.get('type', '')
    remote = request.args.get('remote', '')
    
    # Build query
    query = Job.query.filter_by(status='active')
    
    if search:
        query = query.filter(
            Job.title.ilike(f'%{search}%') | 
            Job.description.ilike(f'%{search}%') |
            Job.company_name.ilike(f'%{search}%')
        )
    
    if location:
        query = query.filter(Job.location.ilike(f'%{location}%'))
    
    if job_type:
        query = query.filter_by(job_type=job_type)
    
    if remote == 'true':
        query = query.filter_by(is_remote=True)
    
    jobs_list = query.order_by(Job.posted_date.desc()).all()
    
    # If user is logged in, calculate match scores
    job_matches = []
    if current_user.is_authenticated and current_user.is_job_seeker():
        for job in jobs_list:
            job_data = {
                'id': job.id,
                'title': job.title,
                'description': job.description,
                'location': job.location,
                'salary_min': job.salary_min,
                'salary_max': job.salary_max
            }
            
            user_data = {
                'skills': current_user.get_skills_list(),
                'experience': current_user.experience or '',
                'preferences': {
                    'locations': current_user.preferred_locations.split(',') if current_user.preferred_locations else [],
                    'salary': current_user.salary_expectation or 0,
                    'remote': current_user.remote_preference
                }
            }
            
            match_score = job_matcher.calculate_match_score(user_data, job_data)
            job_matches.append((job, match_score))
    else:
        job_matches = [(job, None) for job in jobs_list]
    
    return render_template('jobs.html', 
                         job_matches=job_matches,
                         search_query=search,
                         location_query=location,
                         user=current_user)

@app.route('/job/<int:job_id>')
def job_detail(job_id):
    """Job detail page"""
    job = Job.query.get_or_404(job_id)
    
    # Increment view count
    job.increment_views()
    
    # Check if user has applied
    has_applied = False
    if current_user.is_authenticated:
        has_applied = Application.query.filter_by(
            job_id=job_id,
            applicant_id=current_user.id
        ).first() is not None
    
    # Get similar jobs
    similar_jobs = Job.query.filter(
        Job.id != job_id,
        Job.status == 'active'
    ).limit(4).all()
    
    # Calculate match score for current user
    match_score = None
    if current_user.is_authenticated and current_user.is_job_seeker():
        job_data = {
            'id': job.id,
            'title': job.title,
            'description': job.description,
            'location': job.location,
            'salary_min': job.salary_min,
            'salary_max': job.salary_max
        }
        
        user_data = {
            'skills': current_user.get_skills_list(),
            'experience': current_user.experience or '',
            'preferences': {
                'locations': current_user.preferred_locations.split(',') if current_user.preferred_locations else [],
                'salary': current_user.salary_expectation or 0,
                'remote': current_user.remote_preference
            }
        }
        
        match_score = job_matcher.calculate_match_score(user_data, job_data)
    
    return render_template('job_detail.html',
                         job=job,
                         has_applied=has_applied,
                         similar_jobs=similar_jobs,
                         match_score=match_score,
                         user=current_user)

@app.route('/job/<int:job_id>/apply', methods=['POST'])
@login_required
def apply_job(job_id):
    """Apply for a job"""
    job = Job.query.get_or_404(job_id)
    
    # Check if user is job seeker
    if not current_user.is_job_seeker():
        flash('Only job seekers can apply for jobs.', 'danger')
        return redirect(url_for('job_detail', job_id=job_id))
    
    # Check if already applied
    existing_application = Application.query.filter_by(
        job_id=job_id,
        applicant_id=current_user.id
    ).first()
    
    if existing_application:
        flash('You have already applied for this job.', 'warning')
        return redirect(url_for('job_detail', job_id=job_id))
    
    # Calculate match score
    job_data = {
        'id': job.id,
        'title': job.title,
        'description': job.description,
        'location': job.location,
        'salary_min': job.salary_min,
        'salary_max': job.salary_max
    }
    
    user_data = {
        'skills': current_user.get_skills_list(),
        'experience': current_user.experience or '',
        'preferences': {
            'locations': current_user.preferred_locations.split(',') if current_user.preferred_locations else [],
            'salary': current_user.salary_expectation or 0,
            'remote': current_user.remote_preference
        }
    }
    
    match_result = job_matcher.calculate_match_score(user_data, job_data)
    
    # Create application
    application = Application(
        job_id=job_id,
        applicant_id=current_user.id,
        cover_letter=request.form.get('cover_letter', ''),
        resume_text=current_user.resume_text,
        ai_score=match_result['overall_score'],
        ai_feedback=str(match_result)
    )
    
    db.session.add(application)
    db.session.commit()
    
    flash('Application submitted successfully!', 'success')
    return redirect(url_for('job_detail', job_id=job_id))

@app.route('/post-job', methods=['GET', 'POST'])
@employer_required
def post_job():
    """Post a new job"""
    if request.method == 'POST':
        # Get form data
        title = request.form.get('title')
        description = request.form.get('description')
        requirements = request.form.get('requirements')
        company_name = request.form.get('company_name')
        company_website = request.form.get('company_website')
        company_description = request.form.get('company_description')
        job_type = request.form.get('job_type')
        experience_level = request.form.get('experience_level')
        education_required = request.form.get('education_required')
        location = request.form.get('location')
        city = request.form.get('city')
        country = request.form.get('country')
        salary_min = request.form.get('salary_min')
        salary_max = request.form.get('salary_max')
        salary_currency = request.form.get('salary_currency', 'USD')
        
        # Convert salary to integers if provided
        if salary_min:
            salary_min = int(salary_min)
        if salary_max:
            salary_max = int(salary_max)
        
        # Create new job
        job = Job(
            title=title,
            description=description,
            requirements=requirements,
            company_name=company_name,
            company_website=company_website,
            company_description=company_description,
            job_type=job_type,
            experience_level=experience_level,
            education_required=education_required,
            location=location,
            city=city,
            country=country,
            is_remote='is_remote' in request.form,
            is_hybrid='is_hybrid' in request.form,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            posted_by=current_user.id,
            is_featured='is_featured' in request.form,
            expiry_date=datetime.utcnow().replace(year=datetime.utcnow().year + 1)  # Expires in 1 year
        )
        
        db.session.add(job)
        db.session.commit()
        
        flash('Job posted successfully!', 'success')
        return redirect(url_for('job_detail', job_id=job.id))
    
    return render_template('post_job.html', user=current_user)

@app.route('/api/recommendations')
@login_required
def get_recommendations_api():
    """API endpoint for job recommendations"""
    if not current_user.is_job_seeker():
        return jsonify({'error': 'Only job seekers can get recommendations'}), 403
    
    recommendations = job_matcher.get_recommendations(current_user, limit=10)
    
    result = []
    for rec in recommendations:
        job = rec['job']
        match_scores = rec['match_scores']
        
        result.append({
            'job_id': job.id,
            'title': job.title,
            'company': job.company_name,
            'location': job.location,
            'job_type': job.job_type,
            'match_score': match_scores['overall_score'],
            'skills_match': match_scores['skill_match'],
            'matched_skills': match_scores['matched_skills'],
            'missing_skills': match_scores['missing_skills']
        })
    
    return jsonify(result)

@app.route('/api/parse-resume', methods=['POST'])
@login_required
def parse_resume():
    """Parse resume text and extract skills"""
    resume_text = request.form.get('resume_text', '')
    
    if not resume_text:
        return jsonify({'error': 'No resume text provided'}), 400
    
    # Extract skills using the job matcher
    skills = job_matcher.extract_skills(resume_text)
    
    # Update user's resume text
    current_user.resume_text = resume_text
    db.session.commit()
    
    return jsonify({
        'success': True,
        'skills_found': skills,
        'skill_count': len(skills)
    })

# ==================== DATABASE INITIALIZATION ==================== #

def init_db():
    """Initialize database with sample data"""
    with app.app_context():
        # Create tables
        db.create_all()
        
        # Check if we need to create sample data
        if User.query.count() == 0:
            # Create admin user
            admin = User(
                email='admin@example.com',
                first_name='Admin',
                last_name='User',
                user_type='admin'
            )
            admin.set_password('admin123')
            db.session.add(admin)
            
            # Create sample employer
            employer = User(
                email='employer@example.com',
                first_name='Company',
                last_name='Owner',
                user_type='employer',
                headline='Tech Company Founder',
                skills='Python, JavaScript, Management',
                experience='10+ years in tech industry',
                location='San Francisco, CA',
                salary_expectation=150000
            )
            employer.set_password('employer123')
            db.session.add(employer)
            
            # Create sample job seeker
            seeker = User(
                email='seeker@example.com',
                first_name='John',
                last_name='Doe',
                user_type='job_seeker',
                headline='Python Developer',
                skills='Python, Django, Flask, React, AWS',
                experience='5 years as full stack developer',
                education='BS Computer Science',
                location='New York, NY',
                preferred_locations='New York, Remote, San Francisco',
                salary_expectation=120000,
                remote_preference=True,
                resume_text='Experienced Python developer with Django/Flask expertise. Proficient in React and AWS.'
            )
            seeker.set_password('seeker123')
            db.session.add(seeker)
            
            db.session.commit()
            
            # Create sample jobs
            sample_jobs = [
                Job(
                    title='Senior Python Developer',
                    description='We are looking for an experienced Python developer to join our team. You will be responsible for developing and maintaining our web applications using Django and Flask.',
                    requirements='• 5+ years of Python experience\n• Strong knowledge of Django/Flask\n• Experience with React or similar frontend frameworks\n• AWS cloud experience\n• Good understanding of REST APIs',
                    company_name='Tech Solutions Inc.',
                    company_description='A leading technology company specializing in innovative software solutions.',
                    company_website='https://techsolutions.com',
                    job_type='full-time',
                    experience_level='senior',
                    education_required="Bachelor's in Computer Science",
                    location='New York, NY',
                    city='New York',
                    country='USA',
                    is_remote=True,
                    is_hybrid=True,
                    salary_min=120000,
                    salary_max=160000,
                    salary_currency='USD',
                    posted_by=employer.id,
                    is_featured=True,
                    posted_date=datetime.utcnow()
                ),
                Job(
                    title='Frontend Developer (React)',
                    description='Join our frontend team to build amazing user interfaces using React. We work on cutting-edge projects for major clients.',
                    requirements='• 3+ years of React experience\n• Strong JavaScript/TypeScript skills\n• CSS/SCSS expertise\n• Experience with state management (Redux)\n• Git version control',
                    company_name='Web Innovations Ltd.',
                    company_description='Creative web development agency focused on modern technologies.',
                    job_type='full-time',
                    experience_level='mid',
                    education_required="Any technical degree",
                    location='Remote',
                    city='Remote',
                    country='Worldwide',
                    is_remote=True,
                    salary_min=90000,
                    salary_max=130000,
                    salary_currency='USD',
                    posted_by=employer.id,
                    is_featured=True,
                    posted_date=datetime.utcnow()
                ),
                Job(
                    title='Data Scientist',
                    description='Work on machine learning projects and data analysis. Help us build predictive models and extract insights from large datasets.',
                    requirements='• Python and ML libraries (scikit-learn, TensorFlow)\n• Strong statistical background\n• SQL and database experience\n• Data visualization skills\n• 4+ years of experience',
                    company_name='Data Analytics Corp',
                    company_description='Data science consulting firm working with Fortune 500 companies.',
                    job_type='contract',
                    experience_level='senior',
                    education_required="Master's in Data Science or related field",
                    location='San Francisco, CA',
                    city='San Francisco',
                    country='USA',
                    is_remote=False,
                    salary_min=140000,
                    salary_max=180000,
                    salary_currency='USD',
                    posted_by=employer.id,
                    posted_date=datetime.utcnow()
                ),
                Job(
                    title='DevOps Engineer',
                    description='Manage our cloud infrastructure and CI/CD pipelines. Help us build scalable and reliable systems.',
                    requirements='• AWS/Azure/GCP experience\n• Docker and Kubernetes\n• Terraform or similar\n• CI/CD tools (Jenkins, GitLab CI)\n• 4+ years of DevOps experience',
                    company_name='Cloud Systems Inc.',
                    company_description='Cloud infrastructure and DevOps consulting company.',
                    job_type='full-time',
                    experience_level='mid',
                    education_required="Bachelor's degree in IT or related field",
                    location='Austin, TX',
                    city='Austin',
                    country='USA',
                    is_remote=True,
                    is_hybrid=True,
                    salary_min=110000,
                    salary_max=150000,
                    salary_currency='USD',
                    posted_by=employer.id,
                    posted_date=datetime.utcnow()
                ),
                Job(
                    title='Junior Software Engineer',
                    description='Great opportunity for recent graduates to start their career in software development. We provide mentorship and training.',
                    requirements='• Bachelor\'s in Computer Science\n• Basic programming knowledge\n• Willingness to learn\n• Good problem-solving skills\n• Team player attitude',
                    company_name='Startup Innovations',
                    company_description='Fast-growing tech startup in the fintech space.',
                    job_type='full-time',
                    experience_level='entry',
                    education_required="Bachelor's in Computer Science",
                    location='Boston, MA',
                    city='Boston',
                    country='USA',
                    is_remote=False,
                    salary_min=70000,
                    salary_max=90000,
                    salary_currency='USD',
                    posted_by=employer.id,
                    posted_date=datetime.utcnow()
                )
            ]
            
            for job in sample_jobs:
                db.session.add(job)
            
            db.session.commit()
            print("✅ Database initialized with sample data!")
            print("👥 Sample users created:")
            print("   - Admin: admin@example.com / admin123")
            print("   - Employer: employer@example.com / employer123")
            print("   - Job Seeker: seeker@example.com / seeker123")
            print("💼 5 sample jobs created")
        else:
            print("✅ Database already exists with data.")

# ==================== MAIN ==================== #

if __name__ == '__main__':
    init_db()
    print("\n🚀 Starting Smart Job Portal...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("📝 Try logging in with the demo accounts above\n")
    app.run(debug=True, port=5000)