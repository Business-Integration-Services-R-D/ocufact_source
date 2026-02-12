/**
 * API helper functions for communicating with the backend
 */

class APIClient {
    constructor(baseUrl = '/api/proxy') {
        this.baseUrl = baseUrl;
    }

    /**
     * Upload file and run OCR
     */
    async runOCR(file, isPDF = false) {
        const formData = new FormData();
        formData.append('file', file);

        const endpoint = isPDF ? 'ocr/pdf' : 'ocr/image';
        
        try {
            const response = await fetch(`${this.baseUrl}/${endpoint}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`OCR failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('OCR Error:', error);
            throw error;
        }
    }

    /**
     * Run binary detection on uploaded file
     */
    async runBinaryDetection(file, isPDF = false) {
        const formData = new FormData();
        formData.append('file', file);

        const endpoint = isPDF ? 'detect/binary/pdf' : 'detect/binary/image';
        
        try {
            const response = await fetch(`${this.baseUrl}/${endpoint}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Binary detection failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Binary Detection Error:', error);
            throw error;
        }
    }

    /**
     * Run label detection on uploaded file
     */
    async runLabelDetection(file, isPDF = false) {
        const formData = new FormData();
        formData.append('file', file);

        const endpoint = isPDF ? 'detect/labels/pdf' : 'detect/labels/image';
        
        try {
            const response = await fetch(`${this.baseUrl}/${endpoint}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Label detection failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Label Detection Error:', error);
            throw error;
        }
    }

    /**
     * Generate regex from phrases
     */
    async generateRegex(phrases) {
        try {
            const response = await fetch(`${this.baseUrl}/regex/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ phrases })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Regex generation failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Regex Generation Error:', error);
            throw error;
        }
    }

    /**
     * Get stored regex patterns
     */
    async getStoredRegexes() {
        try {
            const response = await fetch(`${this.baseUrl}/regex/stored`);

            if (!response.ok) {
                throw new Error(`Failed to fetch stored regexes: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Get Stored Regexes Error:', error);
            throw error;
        }
    }

    /**
     * Match regex pattern against document
     */
    async matchRegex(file, regex, isPDF = false) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('regex', regex);

        const endpoint = isPDF ? 'regex/match/pdf' : 'regex/match/image';
        
        try {
            const response = await fetch(`${this.baseUrl}/${endpoint}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Regex matching failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Regex Match Error:', error);
            throw error;
        }
    }

    /**
     * Apply masks to document
     */
    async applyMask(file, maskFields, isPDF = false) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('fields', JSON.stringify(maskFields));

        const endpoint = isPDF ? 'mask/pdf' : 'mask/image';
        
        try {
            const response = await fetch(`${this.baseUrl}/${endpoint}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Masking failed: ${response.statusText}`);
            }

            return response.blob();
        } catch (error) {
            console.error('Masking Error:', error);
            throw error;
        }
    }

    /**
     * Match coordinates with OCR text
     */
    async matchCoordinates(file, coordinates) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('coords', JSON.stringify(coordinates));
        
        try {
            const response = await fetch(`${this.baseUrl}/match/coords/image`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Coordinate matching failed: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Coordinate Match Error:', error);
            throw error;
        }
    }

    /**
     * Get model labels from the backend
     */
    async getModelLabels() {
        try {
            const response = await fetch(`${this.baseUrl}/model/labels`);

            if (!response.ok) {
                throw new Error(`Failed to fetch model labels: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Get Model Labels Error:', error);
            throw error;
        }
    }
}

// Utility functions
const utils = {
    /**
     * Format file size for display
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Check if file is PDF
     */
    isPDF(file) {
        return file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
    },

    /**
     * Check if file is image
     */
    isImage(file) {
        return file.type.startsWith('image/') || /\.(jpg|jpeg|png|gif|bmp|webp)$/i.test(file.name);
    },

    /**
     * Validate file type
     */
    validateFileType(file) {
        return this.isPDF(file) || this.isImage(file);
    },

    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    },

    /**
     * Show loading state
     */
    showLoading(element, text = 'Loading...') {
        const originalContent = element.innerHTML;
        element.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status"></span>
            ${text}
        `;
        element.disabled = true;
        
        return () => {
            element.innerHTML = originalContent;
            element.disabled = false;
        };
    },

    /**
     * Download blob as file
     */
    downloadBlob(blob, filename) {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    },

    /**
     * Pretty print JSON
     */
    prettyPrintJSON(obj) {
        return JSON.stringify(obj, null, 2);
    },

    /**
     * Debounce function calls
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// Export for use in other scripts
window.APIClient = APIClient;
window.apiUtils = utils;

