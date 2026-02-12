/**
 * Document viewer with overlay support and Socket.IO integration
 */

class DocumentViewer {
    constructor() {
        this.apiClient = new APIClient();
        this.socket = io();
        this.currentFile = null;
        this.currentResults = null;
        this.binaryResults = null;
        this.labelResults = null;
        this.modelLabels = null;
        this.scale = 1.0;
        this.minScale = 0.1;
        this.maxScale = 5.0;
        this.overlayCanvas = null;
        this.overlayContext = null;
        this.binaryOverlayCanvas = null;
        this.binaryOverlayContext = null;
        this.labelOverlayCanvas = null;
        this.labelOverlayContext = null;
        this.isDragging = false;
        this.dragStart = { x: 0, y: 0 };
        this.viewerOffset = { x: 0, y: 0 };
        this.binaryViewerOffset = { x: 0, y: 0 };
        this.labelViewerOffset = { x: 0, y: 0 };
        
        this.init();
    }

    init() {
        this.setupSocketIO();
        this.setupFileUpload();
        this.setupViewer();
        this.setupControls();
        this.setupTabs();
        this.loadStoredRegexes();
        this.loadModelLabels();
    }

    setupSocketIO() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
        });

        this.socket.on('processing_status', (data) => {
            this.updateProcessingStatus(data);
        });

        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
        });
    }

    setupFileUpload() {
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        // Click to upload
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });

        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFile(e.target.files[0]);
            }
        });

        // Drag and drop
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length > 0) {
                this.handleFile(e.dataTransfer.files[0]);
            }
        });
    }

    setupViewer() {
        // Main viewer
        const viewerContainer = document.getElementById('viewerContainer');
        const overlayCanvas = document.getElementById('overlayCanvas');
        
        this.overlayCanvas = overlayCanvas;
        this.overlayContext = overlayCanvas.getContext('2d');

        // Binary viewer
        const binaryViewerContainer = document.getElementById('binaryViewerContainer');
        const binaryOverlayCanvas = document.getElementById('binaryOverlayCanvas');
        
        this.binaryOverlayCanvas = binaryOverlayCanvas;
        this.binaryOverlayContext = binaryOverlayCanvas.getContext('2d');

        // Label viewer
        const labelViewerContainer = document.getElementById('labelViewerContainer');
        const labelOverlayCanvas = document.getElementById('labelOverlayCanvas');
        
        this.labelOverlayCanvas = labelOverlayCanvas;
        this.labelOverlayContext = labelOverlayCanvas.getContext('2d');

        // Setup mouse events for all viewers
        this.setupViewerEvents(viewerContainer, 'main');
        this.setupViewerEvents(binaryViewerContainer, 'binary');
        this.setupViewerEvents(labelViewerContainer, 'label');

        // Click events for word selection
        overlayCanvas.addEventListener('click', (e) => {
            this.handleCanvasClick(e);
        });
        
        binaryOverlayCanvas.addEventListener('click', (e) => {
            this.handleCanvasClick(e, 'binary');
        });
        
        labelOverlayCanvas.addEventListener('click', (e) => {
            this.handleCanvasClick(e, 'label');
        });
    }

    setupViewerEvents(container, type) {
        // Mouse events for pan and zoom
        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            this.zoom(delta, e.clientX, e.clientY, type);
        });

        // Pan functionality
        container.addEventListener('mousedown', (e) => {
            if (e.button === 0) { // Left mouse button
                this.isDragging = true;
                this.dragStart = { x: e.clientX, y: e.clientY };
                container.style.cursor = 'grabbing';
            }
        });

        document.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const dx = e.clientX - this.dragStart.x;
                const dy = e.clientY - this.dragStart.y;
                
                if (type === 'main') {
                    this.viewerOffset.x += dx;
                    this.viewerOffset.y += dy;
                    this.updateViewerTransform();
                } else if (type === 'binary') {
                    this.binaryViewerOffset.x += dx;
                    this.binaryViewerOffset.y += dy;
                    this.updateViewerTransform('binary');
                } else if (type === 'label') {
                    this.labelViewerOffset.x += dx;
                    this.labelViewerOffset.y += dy;
                    this.updateViewerTransform('label');
                }
                
                this.dragStart = { x: e.clientX, y: e.clientY };
            }

            // Handle hover tooltips
            this.handleMouseMove(e);
        });

        document.addEventListener('mouseup', () => {
            if (this.isDragging) {
                this.isDragging = false;
                container.style.cursor = 'grab';
            }
        });
    }

    setupControls() {
        // Main viewer zoom controls
        document.getElementById('zoomIn').addEventListener('click', () => {
            this.zoom(1.2, null, null, 'main');
        });

        document.getElementById('zoomOut').addEventListener('click', () => {
            this.zoom(0.8, null, null, 'main');
        });

        document.getElementById('resetZoom').addEventListener('click', () => {
            this.resetView('main');
        });

        // Binary viewer zoom controls
        document.getElementById('binaryZoomIn').addEventListener('click', () => {
            this.zoom(1.2, null, null, 'binary');
        });

        document.getElementById('binaryZoomOut').addEventListener('click', () => {
            this.zoom(0.8, null, null, 'binary');
        });

        document.getElementById('binaryResetZoom').addEventListener('click', () => {
            this.resetView('binary');
        });

        // Label viewer zoom controls
        document.getElementById('labelZoomIn').addEventListener('click', () => {
            this.zoom(1.2, null, null, 'label');
        });

        document.getElementById('labelZoomOut').addEventListener('click', () => {
            this.zoom(0.8, null, null, 'label');
        });

        document.getElementById('labelResetZoom').addEventListener('click', () => {
            this.resetView('label');
        });

        // Clear results
        document.getElementById('clearResults').addEventListener('click', () => {
            this.clearResults();
        });

        document.getElementById('showCoordsToggle').addEventListener('change', () => {
            this.toggleCoordsDisplay();
        });

        document.getElementById('clearBinaryResults').addEventListener('click', () => {
            this.clearBinaryResults();
        });

        document.getElementById('clearLabelResults').addEventListener('click', () => {
            this.clearLabelResults();
        });

        document.getElementById('clearRegexResults').addEventListener('click', () => {
            this.clearRegexResults();
        });

        // Event delegation for use regex buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('use-regex-btn')) {
                const regex = e.target.getAttribute('data-regex');
                this.useRegex(regex);
            }
        });

        // Detection buttons
        document.getElementById('runBinaryDetection').addEventListener('click', () => {
            this.runBinaryDetection();
        });

        document.getElementById('runLabelDetection').addEventListener('click', () => {
            this.runLabelDetection();
        });

        // Regex controls
        document.getElementById('generateRegex').addEventListener('click', () => {
            this.generateRegex();
        });

        document.getElementById('testRegex').addEventListener('click', () => {
            this.testRegex();
        });

        // Masking controls
        document.getElementById('previewMask').addEventListener('click', () => {
            this.previewMask();
        });

        document.getElementById('applyMask').addEventListener('click', () => {
            this.applyMask();
        });
    }

    setupTabs() {
        const tabs = document.querySelectorAll('#mainTabs button[data-bs-toggle="tab"]');
        tabs.forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                // Update canvas size when switching tabs
                setTimeout(() => {
                    this.resizeCanvas();
                    this.resizeCanvas('binary');
                    this.resizeCanvas('label');
                }, 100);
            });
        });
    }

    async handleFile(file) {
        if (!apiUtils.validateFileType(file)) {
            apiUtils.showNotification('Please upload a valid image or PDF file', 'danger');
            return;
        }

        this.currentFile = file;
        
        // Show file info
        this.showFileInfo(file);
        
        // Load document in viewer
        await this.loadDocument(file);
        
        // Enable detection buttons
        this.enableDetectionButtons(true);
        
        // Auto-run OCR
        await this.runOCR();
    }

    showFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        
        fileName.textContent = file.name;
        fileSize.textContent = apiUtils.formatFileSize(file.size);
        fileInfo.classList.remove('d-none');
    }

    async loadDocument(file) {
        const viewer = document.getElementById('documentViewer');
        const binaryViewer = document.getElementById('binaryDocumentViewer');
        const labelViewer = document.getElementById('labelDocumentViewer');
        
        if (apiUtils.isPDF(file)) {
            await this.loadPDF(file, viewer);
            await this.loadPDF(file, binaryViewer);
            await this.loadPDF(file, labelViewer);
        } else {
            await this.loadImage(file, viewer);
            await this.loadImage(file, binaryViewer);
            await this.loadImage(file, labelViewer);
        }
        
        this.resizeCanvas();
        this.resizeCanvas('binary');
        this.resizeCanvas('label');
    }

    async loadPDF(file, viewer) {
        try {
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
            
            viewer.innerHTML = '';
            
            // Load first page for now (could be extended for multi-page)
            const page = await pdf.getPage(1);
            const viewport = page.getViewport({ scale: 1.0 });
            
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;
            
            await page.render({
                canvasContext: context,
                viewport: viewport
            }).promise;
            
            viewer.appendChild(canvas);
            viewer.style.cursor = 'grab';
            
        } catch (error) {
            console.error('Error loading PDF:', error);
            apiUtils.showNotification('Error loading PDF', 'danger');
        }
    }

    async loadImage(file, viewer) {
        try {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
            
            img.onload = () => {
                viewer.innerHTML = '';
                viewer.appendChild(img);
                viewer.style.cursor = 'grab';
                URL.revokeObjectURL(img.src);
            };
            
        } catch (error) {
            console.error('Error loading image:', error);
            apiUtils.showNotification('Error loading image', 'danger');
        }
    }

    resizeCanvas(type = 'main') {
        let viewerContainer, canvas, results;
        
        if (type === 'main') {
            viewerContainer = document.getElementById('viewerContainer');
            canvas = this.overlayCanvas;
            results = this.currentResults;
        } else if (type === 'binary') {
            viewerContainer = document.getElementById('binaryViewerContainer');
            canvas = this.binaryOverlayCanvas;
            results = this.binaryResults;
        } else if (type === 'label') {
            viewerContainer = document.getElementById('labelViewerContainer');
            canvas = this.labelOverlayCanvas;
            results = this.labelResults;
        }
        
        if (!canvas || !viewerContainer) return;
        
        const rect = viewerContainer.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        
        // Redraw overlays if we have results
        if (results) {
            if (type === 'binary') {
                this.drawBinaryOverlays(results);
            } else if (type === 'label') {
                this.drawLabelOverlays(results);
            } else {
                this.drawOverlays(results);
            }
        }
    }

    zoom(factor, centerX = null, centerY = null, type = 'main') {
        const newScale = Math.max(this.minScale, Math.min(this.maxScale, this.scale * factor));
        
        if (newScale !== this.scale) {
            // If center point provided, zoom towards that point
            if (centerX !== null && centerY !== null) {
                let containerId, offset;
                
                if (type === 'main') {
                    containerId = 'viewerContainer';
                    offset = this.viewerOffset;
                } else if (type === 'binary') {
                    containerId = 'binaryViewerContainer';
                    offset = this.binaryViewerOffset;
                } else if (type === 'label') {
                    containerId = 'labelViewerContainer';
                    offset = this.labelViewerOffset;
                }
                
                const rect = document.getElementById(containerId).getBoundingClientRect();
                const relativeX = centerX - rect.left;
                const relativeY = centerY - rect.top;
                
                offset.x = relativeX - (relativeX - offset.x) * (newScale / this.scale);
                offset.y = relativeY - (relativeY - offset.y) * (newScale / this.scale);
            }
            
            this.scale = newScale;
            this.updateViewerTransform(type);
        }
    }

    resetView(type = 'main') {
        this.scale = 1.0;
        
        if (type === 'main') {
            this.viewerOffset = { x: 0, y: 0 };
        } else if (type === 'binary') {
            this.binaryViewerOffset = { x: 0, y: 0 };
        } else if (type === 'label') {
            this.labelViewerOffset = { x: 0, y: 0 };
        }
        
        this.updateViewerTransform(type);
    }

    updateViewerTransform(type = 'main') {
        let viewer, offset;
        
        if (type === 'main') {
            viewer = document.getElementById('documentViewer');
            offset = this.viewerOffset;
        } else if (type === 'binary') {
            viewer = document.getElementById('binaryDocumentViewer');
            offset = this.binaryViewerOffset;
        } else if (type === 'label') {
            viewer = document.getElementById('labelDocumentViewer');
            offset = this.labelViewerOffset;
        }
        
        const transform = `translate(${offset.x}px, ${offset.y}px) scale(${this.scale})`;
        viewer.style.transform = transform;
        viewer.style.transformOrigin = '0 0';
        
        // Update overlay canvas transform
        this.resizeCanvas(type);
    }

    async runOCR() {
        if (!this.currentFile) return;
        
        const hideLoading = apiUtils.showLoading(
            document.querySelector('#upload .card-header h5'), 
            'Running OCR...'
        );
        
        try {
            const isPDF = apiUtils.isPDF(this.currentFile);
            const result = await this.apiClient.runOCR(this.currentFile, isPDF);
            
            this.displayResults(result, 'OCR Results');
            
        } catch (error) {
            apiUtils.showNotification(`OCR failed: ${error.message}`, 'danger');
        } finally {
            hideLoading();
        }
    }

    async runBinaryDetection() {
        if (!this.currentFile) return;
        
        const button = document.getElementById('runBinaryDetection');
        const hideLoading = apiUtils.showLoading(button, 'Detecting...');
        
        try {
            const isPDF = apiUtils.isPDF(this.currentFile);
            const result = await this.apiClient.runBinaryDetection(this.currentFile, isPDF);
            
            this.binaryResults = result;
            this.displayBinaryResults(result, 'Binary Detection Results');
            this.drawBinaryOverlays(result);
            
        } catch (error) {
            apiUtils.showNotification(`Binary detection failed: ${error.message}`, 'danger');
        } finally {
            hideLoading();
        }
    }

    async runLabelDetection() {
        if (!this.currentFile) return;
        
        const button = document.getElementById('runLabelDetection');
        const hideLoading = apiUtils.showLoading(button, 'Detecting...');
        
        try {
            const isPDF = apiUtils.isPDF(this.currentFile);
            const result = await this.apiClient.runLabelDetection(this.currentFile, isPDF);
            
            this.labelResults = result;
            this.displayLabelResults(result, 'Label Detection Results');
            this.drawLabelOverlays(result);
            
        } catch (error) {
            apiUtils.showNotification(`Label detection failed: ${error.message}`, 'danger');
        } finally {
            hideLoading();
        }
    }

    drawBinaryOverlays(results) {
        const ctx = this.binaryOverlayContext;
        ctx.clearRect(0, 0, this.binaryOverlayCanvas.width, this.binaryOverlayCanvas.height);
        
        // Note: Binary overlays will need coordinates from OCR results
        // For now, we'll skip drawing overlays since we only have text and label
        // The overlays would need to be drawn based on the original OCR coordinates
        console.log('Binary results (overlays disabled - no coordinates):', results);
    }

    drawLabelOverlays(results) {
        const ctx = this.labelOverlayContext;
        ctx.clearRect(0, 0, this.labelOverlayCanvas.width, this.labelOverlayCanvas.height);
        
        // Note: Label overlays will need coordinates from OCR results
        // For now, we'll skip drawing overlays since we only have text and label
        // The overlays would need to be drawn based on the original OCR coordinates
        console.log('Label results (overlays disabled - no coordinates):', results);
    }

    drawWordHighlight(coords, fillColor, borderColor, type = 'main') {
        let ctx, offset;
        
        if (type === 'main') {
            ctx = this.overlayContext;
            offset = this.viewerOffset;
        } else if (type === 'binary') {
            ctx = this.binaryOverlayContext;
            offset = this.binaryViewerOffset;
        } else if (type === 'label') {
            ctx = this.labelOverlayContext;
            offset = this.labelViewerOffset;
        }
        
        const [x1, y1, x2, y2] = coords;
        
        // Apply current transform
        const rect = this.getTransformedRect(x1, y1, x2, y2, type);
        
        // Draw filled rectangle
        ctx.fillStyle = fillColor;
        ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
        
        // Draw border
        ctx.strokeStyle = borderColor;
        ctx.lineWidth = 2;
        ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
    }

    getTransformedRect(x1, y1, x2, y2, type = 'main') {
        // Transform coordinates based on current scale and offset
        let viewer, viewerContainer, offset;
        
        if (type === 'main') {
            viewer = document.getElementById('documentViewer');
            viewerContainer = document.getElementById('viewerContainer');
            offset = this.viewerOffset;
        } else if (type === 'binary') {
            viewer = document.getElementById('binaryDocumentViewer');
            viewerContainer = document.getElementById('binaryViewerContainer');
            offset = this.binaryViewerOffset;
        } else if (type === 'label') {
            viewer = document.getElementById('labelDocumentViewer');
            viewerContainer = document.getElementById('labelViewerContainer');
            offset = this.labelViewerOffset;
        }
        
        const viewerRect = viewer.getBoundingClientRect();
        const containerRect = viewerContainer.getBoundingClientRect();
        
        const scaleX = this.scale;
        const scaleY = this.scale;
        
        return {
            x: (x1 * scaleX) + offset.x + (viewerRect.left - containerRect.left),
            y: (y1 * scaleY) + offset.y + (viewerRect.top - containerRect.top),
            width: (x2 - x1) * scaleX,
            height: (y2 - y1) * scaleY
        };
    }


    handleMouseMove(e) {
        if (!this.currentResults) return;
        
        const rect = this.overlayCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Find word under cursor
        const word = this.findWordAtPosition(x, y);
        
        if (word) {
            this.showTooltip(e.clientX, e.clientY, word);
        } else {
            this.hideTooltip();
        }
    }

    findWordAtPosition(x, y) {
        if (!this.currentResults) return null;
        
        for (const page of this.currentResults.pages) {
            for (const word of page.words) {
                const rect = this.getTransformedRect(...word.coords);
                if (x >= rect.x && x <= rect.x + rect.width &&
                    y >= rect.y && y <= rect.y + rect.height) {
                    return word;
                }
            }
        }
        
        return null;
    }

    showTooltip(x, y, word) {
        const tooltip = document.getElementById('tooltip');
        const content = tooltip.querySelector('.tooltip-content');
        
        let tooltipText = `Text: "${word.text}"<br>Confidence: ${(word.confidence * 100).toFixed(1)}%`;
        
        if (word.is_sensitive !== undefined) {
            tooltipText += `<br>Sensitive: ${word.is_sensitive ? 'Yes' : 'No'}`;
        }
        
        if (word.label) {
            tooltipText += `<br>Label: ${word.label}`;
        }
        
        if (word.score !== undefined) {
            tooltipText += `<br>Score: ${(word.score * 100).toFixed(1)}%`;
        }
        
        content.innerHTML = tooltipText;
        
        tooltip.style.left = (x + 10) + 'px';
        tooltip.style.top = (y - 10) + 'px';
        tooltip.classList.remove('d-none');
    }

    hideTooltip() {
        document.getElementById('tooltip').classList.add('d-none');
    }

    handleCanvasClick(e) {
        const rect = this.overlayCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const word = this.findWordAtPosition(x, y);
        
        if (word) {
            this.showWordDetails(word);
        }
    }

    showWordDetails(word) {
        // Create modal or detailed view for word
        const modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Word Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <p><strong>Text:</strong> "${word.text}"</p>
                        <p><strong>Coordinates:</strong> [${word.coords.join(', ')}]</p>
                        <p><strong>OCR Confidence:</strong> ${(word.confidence * 100).toFixed(2)}%</p>
                        ${word.is_sensitive !== undefined ? `<p><strong>Sensitive:</strong> ${word.is_sensitive ? 'Yes' : 'No'}</p>` : ''}
                        ${word.label ? `<p><strong>Label:</strong> ${word.label}</p>` : ''}
                        ${word.score !== undefined ? `<p><strong>Detection Score:</strong> ${(word.score * 100).toFixed(2)}%</p>` : ''}
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        const bootstrapModal = new bootstrap.Modal(modal);
        bootstrapModal.show();
        
        modal.addEventListener('hidden.bs.modal', () => {
            document.body.removeChild(modal);
        });
    }

    async generateRegex() {
        const textarea = document.getElementById('regexPhrases');
        const phrases = textarea.value.split('\n').filter(p => p.trim()).map(p => p.trim());
        
        if (phrases.length < 8) {
            apiUtils.showNotification('Please enter at least 8 phrases', 'warning');
            return;
        }
        
        const button = document.getElementById('generateRegex');
        const hideLoading = apiUtils.showLoading(button, 'Generating...');
        
        try {
            const result = await this.apiClient.generateRegex(phrases);
            
            apiUtils.showNotification(`Regex generated with ${(result.confidence * 100).toFixed(1)}% confidence`, 'success');
            
            // Add to custom regex field
            document.getElementById('customRegex').value = result.regex;
            
            // Refresh stored regexes
            await this.loadStoredRegexes();
            
        } catch (error) {
            apiUtils.showNotification(`Regex generation failed: ${error.message}`, 'danger');
        } finally {
            hideLoading();
        }
    }

    async loadStoredRegexes() {
        try {
            const result = await this.apiClient.getStoredRegexes();
            const container = document.getElementById('storedRegexList');
            
            if (result.regexes && result.regexes.length > 0) {
                container.innerHTML = result.regexes.map((regex, index) => `
                    <div class="border rounded p-2 mb-2">
                        <div class="d-flex justify-content-between align-items-start">
                            <div class="flex-grow-1">
                                <code class="small">${regex.regex}</code>
                                <div class="text-muted small">Confidence: ${(regex.confidence * 100).toFixed(1)}%</div>
                            </div>
                            <button class="btn btn-sm btn-outline-primary ms-2 use-regex-btn" data-regex="${regex.regex.replace(/"/g, '&quot;')}">
                                Use
                            </button>
                        </div>
                    </div>
                `).join('');
            } else {
                container.innerHTML = '<p class="text-muted">No stored patterns yet</p>';
            }
            
        } catch (error) {
            console.error('Error loading stored regexes:', error);
        }
    }

    async loadModelLabels() {
        try {
            const result = await this.apiClient.getModelLabels();
            this.modelLabels = result;
            
            // Update the label legend with actual model labels
            this.updateLabelLegend(result.label_labels);
            
            console.log('Model labels loaded:', result);
            
        } catch (error) {
            console.error('Error loading model labels:', error);
            // Fallback to default labels if loading fails
            this.modelLabels = {
                label_labels: {
                    "0": "O",
                    "1": "B-NAME",
                    "2": "I-NAME",
                    "3": "B-EMAIL",
                    "4": "I-EMAIL",
                    "5": "B-MOBILE",
                    "6": "I-MOBILE",
                    "7": "B-ID_NO",
                    "8": "I-ID_NO"
                }
            };
            this.updateLabelLegend(this.modelLabels.label_labels);
        }
    }

    updateLabelLegend(labelLabels) {
        const legendContainer = document.getElementById('labelLegend');
        
        if (!labelLabels || Object.keys(labelLabels).length === 0) {
            legendContainer.innerHTML = '<p class="text-muted">No labels available</p>';
            return;
        }

        // Generate colors for different label types
        const colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f', '#bb8fce', 
            '#85c1e9', '#f8c471', '#a8e6cf', '#ffd3a5', '#fd79a8'
        ];
        
        // Extract unique label types (excluding O and BIO prefixes)
        const labelTypes = new Set();
        Object.values(labelLabels).forEach(label => {
            if (label !== 'O') {
                // Remove B- and I- prefixes to get the base label type
                const baseLabel = label.replace(/^[BI]-/, '');
                labelTypes.add(baseLabel);
            }
        });

        // Create legend items
        const legendItems = Array.from(labelTypes).map((labelType, index) => {
            const color = colors[index % colors.length];
            return `
                <div class="legend-item d-flex align-items-center mb-2">
                    <div class="legend-color me-2" style="width: 20px; height: 20px; border-radius: 3px; background-color: ${color};"></div>
                    <span>${labelType}</span>
                </div>
            `;
        }).join('');

        legendContainer.innerHTML = legendItems || '<p class="text-muted">No labels available</p>';
    }

    useRegex(regex) {
        document.getElementById('customRegex').value = regex;
        apiUtils.showNotification('Regex pattern loaded into test field', 'success');
    }

    async testRegex() {
        const regex = document.getElementById('customRegex').value.trim();
        
        if (!regex) {
            apiUtils.showNotification('Please enter a regex pattern', 'warning');
            return;
        }
        
        if (!this.currentFile) {
            apiUtils.showNotification('Please upload a document first', 'warning');
            return;
        }
        
        const button = document.getElementById('testRegex');
        const hideLoading = apiUtils.showLoading(button, 'Testing...');
        
        try {
            const isPDF = apiUtils.isPDF(this.currentFile);
            const result = await this.apiClient.matchRegex(this.currentFile, regex, isPDF); // { matches: string[] }
            
            this.currentResults = result;
            this.displayRegexResults(result, 'Regex Match Results');
            // Overlays not drawn for simple matches array
            
        } catch (error) {
            apiUtils.showNotification(`Regex test failed: ${error.message}`, 'danger');
        } finally {
            hideLoading();
        }
    }

    // drawRegexOverlays removed: matches endpoint now returns simple list

    async previewMask() {
        if (!this.currentResults) {
            apiUtils.showNotification('Please run detection first', 'warning');
            return;
        }
        
        // Create preview overlay
        this.drawMaskPreview();
        apiUtils.showNotification('Mask preview applied', 'info');
    }

    drawMaskPreview() {
        const ctx = this.overlayContext;
        ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        
        if (!this.currentResults) return;
        
        this.currentResults.pages.forEach(page => {
            page.words.forEach(word => {
                if (word.is_sensitive || (word.label && word.label !== 'O')) {
                    // Use semi-transparent mask
                    this.drawWordHighlight(word.coords, '#00000040', '#000000');
                }
            });
        });
    }

    async applyMask() {
        if (!this.currentResults || !this.currentFile) {
            apiUtils.showNotification('Please run detection first', 'warning');
            return;
        }
        
        const button = document.getElementById('applyMask');
        const hideLoading = apiUtils.showLoading(button, 'Masking...');
        
        try {
            // Create mask fields from current results
            const maskFields = [];
            
            this.currentResults.pages.forEach(page => {
                page.words.forEach(word => {
                    if (word.is_sensitive || (word.label && word.label !== 'O')) {
                        maskFields.push({
                            coords: word.coords,
                            color: this.getMaskColor(word)
                        });
                    }
                });
            });
            
            const isPDF = apiUtils.isPDF(this.currentFile);
            const maskedBlob = await this.apiClient.applyMask(this.currentFile, maskFields, isPDF);
            
            // Download masked file
            const extension = isPDF ? 'pdf' : 'png';
            const filename = `masked_${this.currentFile.name.split('.')[0]}.${extension}`;
            
            apiUtils.downloadBlob(maskedBlob, filename);
            apiUtils.showNotification('Masked document downloaded', 'success');
            
        } catch (error) {
            apiUtils.showNotification(`Masking failed: ${error.message}`, 'danger');
        } finally {
            hideLoading();
        }
    }

    getMaskColor(word) {
        if (word.label) {
            // Generate colors for different label types
            const colors = [
                '#ff6b6b', '#4ecdc4', '#45b7d1', '#f7dc6f', '#bb8fce', 
                '#85c1e9', '#f8c471', '#a8e6cf', '#ffd3a5', '#fd79a8'
            ];
            
            // Create label color mapping
            const labelColors = {};
            if (this.modelLabels && this.modelLabels.label_labels) {
                const labelTypes = new Set();
                Object.values(this.modelLabels.label_labels).forEach(label => {
                    if (label !== 'O') {
                        const baseLabel = label.replace(/^[BI]-/, '');
                        labelTypes.add(baseLabel);
                    }
                });
                
                Array.from(labelTypes).forEach((labelType, index) => {
                    labelColors[labelType] = colors[index % colors.length];
                });
            }
            
            return labelColors[word.label] || '#ff0000';
        }
        return document.getElementById('sensitiveColor').value;
    }

    displayResults(results, title) {
        const output = document.getElementById('resultsOutput');
        const showCoords = document.getElementById('showCoordsToggle').checked;
        
        if (showCoords) {
            // Show full results with coordinates
            output.textContent = `${title}:\n\n${apiUtils.prettyPrintJSON(results)}`;
        } else {
            // Show clean text only
            const cleanText = this.organizeTextByLines(results);
            output.textContent = `${title}:\n\n${cleanText}`;
        }
    }

    organizeTextByLines(results) {
        try {
            if (!results || !results.pages || results.pages.length === 0) {
                return "No text found";
            }

            const allWords = [];
            results.pages.forEach(page => {
                page.words.forEach(word => {
                    if (word.text && word.text.trim()) {
                        allWords.push({
                            text: word.text.trim(),
                            coords: word.coords,
                            confidence: word.confidence
                        });
                    }
                });
            });

            if (allWords.length === 0) {
                return "No text found";
            }

            // Sort words by Y coordinate (top to bottom), then by X coordinate (left to right)
            allWords.sort((a, b) => {
                const aY = a.coords[1]; // minY
                const bY = b.coords[1]; // minY
                if (Math.abs(aY - bY) > 20) { // Different lines (20px tolerance)
                    return aY - bY;
                }
                // Same line, sort by X coordinate
                return a.coords[0] - b.coords[0]; // minX
            });

            // Group words into lines
            const lines = [];
            let currentLine = [];
            let currentLineY = null;
            const lineTolerance = 20;

            allWords.forEach(word => {
                const wordY = word.coords[1]; // minY
                
                if (currentLineY === null || Math.abs(wordY - currentLineY) <= lineTolerance) {
                    // Same line
                    currentLine.push(word.text);
                    if (currentLineY === null) {
                        currentLineY = wordY;
                    }
                } else {
                    // New line
                    if (currentLine.length > 0) {
                        lines.push(currentLine.join(' '));
                    }
                    currentLine = [word.text];
                    currentLineY = wordY;
                }
            });

            // Add the last line
            if (currentLine.length > 0) {
                lines.push(currentLine.join(' '));
            }

            return lines.join('\n');
        } catch (error) {
            console.error('Error organizing text by lines:', error);
            // Fallback: just join all text
            const allText = results.pages.flatMap(page => 
                page.words.map(word => word.text).filter(text => text && text.trim())
            );
            return allText.join(' ');
        }
    }

    toggleCoordsDisplay() {
        // Re-display current results with new format
        if (this.currentResults) {
            this.displayResults(this.currentResults, 'OCR Results');
        }
    }

    displayBinaryResults(results, title) {
        const output = document.getElementById('binaryResultsOutput');
        
        // Check if any sensitive data was found
        const totalSensitiveWords = results.pages.reduce((total, page) => total + page.words.length, 0);
        
        if (totalSensitiveWords === 0) {
            output.textContent = `${title}:\n\nNo sensitive data detected.`;
        } else {
            output.textContent = `${title}:\n\n${apiUtils.prettyPrintJSON(results)}`;
        }
    }

    displayLabelResults(results, title) {
        const output = document.getElementById('labelResultsOutput');
        
        // Check if any sensitive data was found
        const totalSensitiveWords = results.pages.reduce((total, page) => total + page.words.length, 0);
        
        if (totalSensitiveWords === 0) {
            output.textContent = `${title}:\n\nNo sensitive data detected.`;
        } else {
            output.textContent = `${title}:\n\n${apiUtils.prettyPrintJSON(results)}`;
        }
    }

    clearResults() {
        document.getElementById('resultsOutput').textContent = 'No results yet';
        this.currentResults = null;
        
        if (this.overlayContext) {
            this.overlayContext.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        }
        
    }

    clearBinaryResults() {
        document.getElementById('binaryResultsOutput').textContent = 'No results yet';
        this.binaryResults = null;
        
        if (this.binaryOverlayContext) {
            this.binaryOverlayContext.clearRect(0, 0, this.binaryOverlayCanvas.width, this.binaryOverlayCanvas.height);
        }
    }

    clearLabelResults() {
        document.getElementById('labelResultsOutput').textContent = 'No results yet';
        this.labelResults = null;
        
        if (this.labelOverlayContext) {
            this.labelOverlayContext.clearRect(0, 0, this.labelOverlayCanvas.width, this.labelOverlayCanvas.height);
        }
    }

    displayRegexResults(results, title) {
        const output = document.getElementById('regexResultsOutput');
        const matches = (results && Array.isArray(results.matches)) ? results.matches : [];
        
        if (matches.length === 0) {
            output.textContent = `${title}:\n\nNo matches found.`;
        } else {
            output.textContent = `${title}:\n\n${apiUtils.prettyPrintJSON(matches)}`;
        }
    }

    clearRegexResults() {
        document.getElementById('regexResultsOutput').textContent = 'No regex test results yet';
        this.currentResults = null;
        
        if (this.overlayContext) {
            this.overlayContext.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        }
    }

    enableDetectionButtons(enabled) {
        document.getElementById('runBinaryDetection').disabled = !enabled;
        document.getElementById('runLabelDetection').disabled = !enabled;
        document.getElementById('previewMask').disabled = !enabled;
        document.getElementById('applyMask').disabled = !enabled;
    }

    updateProcessingStatus(data) {
        const statusCard = document.getElementById('statusCard');
        const statusMessage = document.getElementById('statusMessage');
        const progressBar = document.getElementById('progressBar');
        
        statusCard.classList.remove('d-none');
        statusMessage.textContent = data.message;
        
        // Update progress based on status
        let progress = 0;
        switch (data.status) {
            case 'started': progress = 10; break;
            case 'ocr': progress = 30; break;
            case 'detection': progress = 70; break;
            case 'completed': progress = 100; break;
            case 'error': progress = 0; break;
        }
        
        progressBar.style.width = progress + '%';
        
        if (data.status === 'completed' || data.status === 'error') {
            setTimeout(() => {
                statusCard.classList.add('d-none');
            }, 3000);
        }
    }
}

// Initialize viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.viewer = new DocumentViewer();
});

// Handle window resize
window.addEventListener('resize', apiUtils.debounce(() => {
    if (window.viewer) {
        window.viewer.resizeCanvas();
        window.viewer.resizeCanvas('binary');
        window.viewer.resizeCanvas('label');
    }
}, 250));

