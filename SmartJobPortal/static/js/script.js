// Main JavaScript for Smart Job Portal

$(document).ready(function() {
    // Initialize tooltips
    $('[data-bs-toggle="tooltip"]').tooltip();
    
    // Initialize popovers
    $('[data-bs-toggle="popover"]').popover();
    
    // Auto-dismiss alerts after 5 seconds
    setTimeout(function() {
        $('.alert').alert('close');
    }, 5000);
    
    // Form validation
    $('form').on('submit', function() {
        $(this).find('button[type="submit"]').prop('disabled', true);
    });
    
    // Email validation
    $('input[type="email"]').on('blur', function() {
        const email = $(this).val();
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        
        if (email && !emailRegex.test(email)) {
            $(this).addClass('is-invalid');
            $(this).next('.invalid-feedback').text('Please enter a valid email address.');
        } else {
            $(this).removeClass('is-invalid');
        }
    });
    
    // Password strength indicator
    $('#password').on('keyup', function() {
        const password = $(this).val();
        let strength = 0;
        
        if (password.length >= 8) strength++;
        if (/[A-Z]/.test(password)) strength++;
        if (/[0-9]/.test(password)) strength++;
        if (/[^A-Za-z0-9]/.test(password)) strength++;
        
        const strengthText = ['Very Weak', 'Weak', 'Fair', 'Good', 'Strong'][strength];
        const strengthColor = ['danger', 'warning', 'info', 'primary', 'success'][strength];
        
        $('#password-strength').text(strengthText).removeClass().addClass('badge bg-' + strengthColor);
    });
    
    // Job search autocomplete
    $('#jobSearch').on('keyup', function() {
        const query = $(this).val();
        if (query.length >= 2) {
            $.get('/api/jobs/autocomplete', { q: query }, function(data) {
                // Implement autocomplete dropdown
            });
        }
    });
    
    // Resume upload preview
    $('#resumeUpload').on('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const fileName = file.name;
            const fileSize = (file.size / 1024 / 1024).toFixed(2); // MB
            
            if (fileSize > 5) {
                alert('File size should be less than 5MB');
                $(this).val('');
                return;
            }
            
            $('#resumeFileName').text(fileName + ' (' + fileSize + ' MB)');
            
            // Parse resume using AJAX
            parseResume(file);
        }
    });
    
    // Skill input with tags
    $('#skillsInput').on('keypress', function(e) {
        if (e.which === 13 || e.which === 44) {
            e.preventDefault();
            const skill = $(this).val().trim();
            if (skill) {
                addSkillTag(skill);
                $(this).val('');
            }
        }
    });
    
    // Initialize date pickers
    $('.datepicker').datepicker({
        format: 'yyyy-mm-dd',
        autoclose: true
    });
    
    // AJAX form submissions
    $('.ajax-form').on('submit', function(e) {
        e.preventDefault();
        
        const form = $(this);
        const formData = new FormData(this);
        
        $.ajax({
            url: form.attr('action'),
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                if (response.success) {
                    showNotification('Success!', response.message, 'success');
                } else {
                    showNotification('Error!', response.message, 'error');
                }
            },
            error: function() {
                showNotification('Error!', 'Something went wrong. Please try again.', 'error');
            }
        });
    });
    
    // Load more jobs
    let currentPage = 1;
    $('#loadMoreJobs').on('click', function() {
        currentPage++;
        const $button = $(this);
        $button.prop('disabled', true).html('<span class="spinner-border spinner-border-sm"></span> Loading...');
        
        $.get('/api/jobs', { page: currentPage }, function(data) {
            if (data.jobs.length > 0) {
                // Append new jobs
                // ...
                $button.prop('disabled', false).text('Load More');
            } else {
                $button.remove();
            }
        });
    });
    
    // Real-time notification check
    if (window.Notification && Notification.permission === 'granted') {
        setInterval(checkNotifications, 30000); // Check every 30 seconds
    }
});

// Helper Functions

function addSkillTag(skill) {
    const tag = `<span class="skill-tag">${skill} <button type="button" class="btn-close btn-close-white" onclick="removeSkillTag(this)"></button></span>`;
    $('#skillTags').append(tag);
    
    // Update hidden input
    const skills = [];
    $('#skillTags .skill-tag').each(function() {
        skills.push($(this).text().trim().replace('×', ''));
    });
    $('#skills').val(skills.join(','));
}

function removeSkillTag(button) {
    $(button).parent().remove();
    
    // Update hidden input
    const skills = [];
    $('#skillTags .skill-tag').each(function() {
        skills.push($(this).text().trim().replace('×', ''));
    });
    $('#skills').val(skills.join(','));
}

function parseResume(file) {
    const formData = new FormData();
    formData.append('resume', file);
    
    $.ajax({
        url: '/api/parse-resume',
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(data) {
            if (data.success) {
                // Auto-fill form fields
                if (data.name) $('#fullName').val(data.name);
                if (data.email) $('#email').val(data.email);
                if (data.phone) $('#phone').val(data.phone);
                if (data.skills) {
                    data.skills.forEach(skill => addSkillTag(skill));
                }
                
                showNotification('Success!', 'Resume parsed successfully.', 'success');
            }
        },
        error: function() {
            showNotification('Error!', 'Failed to parse resume.', 'error');
        }
    });
}

function showNotification(title, message, type) {
    // Create notification element
    const notification = $(`
        <div class="alert alert-${type} alert-dismissible fade show" role="alert">
            <strong>${title}</strong> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    
    // Add to page
    $('.notifications').prepend(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => notification.alert('close'), 5000);
}

function checkNotifications() {
    $.get('/api/notifications/unread', function(data) {
        if (data.count > 0) {
            // Show notification badge
            $('#notificationBadge').text(data.count).show();
            
            // Request browser notification permission
            if (Notification.permission === 'granted') {
                new Notification('New Notification', {
                    body: `You have ${data.count} new notifications`,
                    icon: '/static/favicon.ico'
                });
            }
        }
    });
}

// Request notification permission
function requestNotificationPermission() {
    if (window.Notification && Notification.permission === 'default') {
        Notification.requestPermission().then(function(permission) {
            if (permission === 'granted') {
                console.log('Notification permission granted');
            }
        });
    }
}

// Smooth scroll to element
function scrollToElement(elementId) {
    $('html, body').animate({
        scrollTop: $(elementId).offset().top - 70
    }, 1000);
}

// Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Calculate days ago
function daysAgo(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now - date);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    return `${Math.floor(diffDays / 365)} years ago`;
}

// Initialize when page loads
$(window).on('load', function() {
    // Request notification permission
    requestNotificationPermission();
    
    // Animate elements on scroll
    $(window).on('scroll', function() {
        $('.animate-on-scroll').each(function() {
            const elementTop = $(this).offset().top;
            const elementBottom = elementTop + $(this).outerHeight();
            const viewportTop = $(window).scrollTop();
            const viewportBottom = viewportTop + $(window).height();
            
            if (elementBottom > viewportTop && elementTop < viewportBottom) {
                $(this).addClass('animated');
            }
        });
    });
    
    // Trigger initial scroll check
    $(window).trigger('scroll');
});