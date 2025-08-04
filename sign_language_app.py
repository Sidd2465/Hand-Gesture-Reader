import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Custom Sign Language Recognition",
    page_icon="🤟",
    layout="wide"
)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class CustomSignLanguageRecognizer:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.model = None
        self.custom_actions = {}
        self.training_data = []
        
    def extract_landmarks(self, image):
        """Extract hand landmarks from image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
        
        return landmarks, results
    
    def normalize_landmarks(self, landmarks):
        """Normalize landmarks relative to wrist position"""
        if len(landmarks) < 63:  # 21 landmarks * 3 coordinates
            return [0] * 63
        
        landmarks = np.array(landmarks).reshape(-1, 3)
        
        # Normalize relative to wrist (landmark 0)
        if len(landmarks) > 0:
            wrist = landmarks[0]
            landmarks = landmarks - wrist
            
            # Scale by hand size (distance from wrist to middle finger tip)
            if len(landmarks) >= 12:  # Ensure we have middle finger tip
                hand_size = np.linalg.norm(landmarks[12] - landmarks[0])
                if hand_size > 0:
                    landmarks = landmarks / hand_size
        
        return landmarks.flatten()
    
    def add_training_sample(self, action_name, landmarks):
        """Add a training sample for a custom action"""
        normalized_landmarks = self.normalize_landmarks(landmarks)
        self.training_data.append({
            'action': action_name,
            'landmarks': normalized_landmarks,
            'timestamp': datetime.now().isoformat()
        })
        
        if action_name not in self.custom_actions:
            self.custom_actions[action_name] = 0
        self.custom_actions[action_name] += 1
    
    def train_model(self):
        """Train the model with collected training data"""
        if len(self.training_data) < 2:
            return False, "Need at least 2 training samples"
        
        # Prepare training data
        X = []
        y = []
        action_names = list(self.custom_actions.keys())
        
        for sample in self.training_data:
            X.append(sample['landmarks'])
            y.append(action_names.index(sample['action']))
        
        X = np.array(X)
        y = np.array(y)
        
        # Check if we have enough samples per class
        unique_classes, counts = np.unique(y, return_counts=True)
        if np.min(counts) < 2:
            return False, "Each action needs at least 2 training samples"
        
        # Train model
        if len(X) > 4:  # Only use train/test split if we have enough data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        
        self.model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        self.model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return True, f"Model trained successfully! Accuracy: {accuracy:.2%}"
    
    def predict_action(self, landmarks):
        """Predict action from landmarks"""
        if self.model is None or len(self.custom_actions) == 0:
            return None, 0.0
        
        if len(landmarks) == 0:
            return None, 0.0
        
        normalized_landmarks = self.normalize_landmarks(landmarks)
        
        # Ensure we have the right number of features
        if len(normalized_landmarks) != 63:
            normalized_landmarks = normalized_landmarks[:63] if len(normalized_landmarks) > 63 else normalized_landmarks + [0] * (63 - len(normalized_landmarks))
        
        try:
            # Predict
            prediction_idx = self.model.predict([normalized_landmarks])[0]
            confidence = np.max(self.model.predict_proba([normalized_landmarks]))
            
            action_names = list(self.custom_actions.keys())
            predicted_action = action_names[prediction_idx]
            
            return predicted_action, confidence
        except:
            return None, 0.0
    
    def export_model(self):
        """Export model and training data"""
        export_data = {
            'custom_actions': self.custom_actions,
            'training_data': self.training_data,
            'model_trained': self.model is not None
        }
        return json.dumps(export_data, indent=2)
    
    def import_model(self, import_data):
        """Import model and training data"""
        try:
            data = json.loads(import_data)
            self.custom_actions = data.get('custom_actions', {})
            self.training_data = data.get('training_data', [])
            
            if data.get('model_trained', False) and len(self.training_data) > 0:
                success, message = self.train_model()
                return success, message
            return True, "Data imported successfully"
        except Exception as e:
            return False, f"Import failed: {str(e)}"

def main():
    st.title("🤟 Custom Sign Language Recognition App")
    st.markdown("### Create and Train Your Own Sign Language Actions")
    
    # Initialize recognizer
    if 'recognizer' not in st.session_state:
        st.session_state.recognizer = CustomSignLanguageRecognizer()
        st.session_state.training_mode = False
        st.session_state.current_action = ""
        st.session_state.prediction_history = []
    
    # Sidebar
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode", ["🎯 Recognition", "📚 Training", "⚙️ Model Management"])
    
    if mode == "📚 Training":
        training_mode()
    elif mode == "⚙️ Model Management":
        model_management_mode()
    else:
        recognition_mode()

def training_mode():
    st.header("Training Mode")
    st.markdown("Create custom actions by recording multiple examples of each gesture.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Record Training Samples")
        
        # Action name input
        action_name = st.text_input("Action Name", placeholder="e.g., 'Hello', 'Thank You', 'Yes', etc.")
        
        if not action_name:
            st.warning("Please enter an action name to start training.")
            return
        
        # Camera input for training
        training_image = st.camera_input(f"Record sample for: {action_name}")
        
        if training_image is not None:
            # Convert to OpenCV format
            bytes_data = training_image.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Process image
            landmarks, results = st.session_state.recognizer.extract_landmarks(cv2_img)
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        cv2_img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                st.image(cv2_img, channels="BGR", caption="Training Sample")
                
                # Add training sample button
                if st.button("✅ Add This Sample", type="primary"):
                    st.session_state.recognizer.add_training_sample(action_name, landmarks)
                    st.success(f"Sample added for '{action_name}'!")
                    st.rerun()
            else:
                st.warning("No hand detected. Make sure your hand is clearly visible.")
    
    with col2:
        st.subheader("Training Progress")
        
        # Display current actions and sample counts
        if st.session_state.recognizer.custom_actions:
            st.write("**Recorded Actions:**")
            for action, count in st.session_state.recognizer.custom_actions.items():
                st.write(f"• {action}: {count} samples")
                
                # Progress bar for each action
                progress = min(count / 5, 1.0)  # Recommend 5+ samples per action
                st.progress(progress)
                
                if count < 3:
                    st.caption("⚠️ Needs more samples (min 3 recommended)")
                elif count < 5:
                    st.caption("⚡ Good, more samples will improve accuracy")
                else:
                    st.caption("✅ Well trained!")
        else:
            st.info("No actions recorded yet. Start by taking some training samples!")
        
        st.markdown("---")
        
        # Train model button
        if len(st.session_state.recognizer.custom_actions) >= 2:
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    success, message = st.session_state.recognizer.train_model()
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
        else:
            st.info("Create at least 2 different actions to train the model.")
        
        # Training tips
        st.subheader("Training Tips")
        st.markdown("""
        **For best results:**
        - Record 5-10 samples per action
        - Vary hand position slightly between samples
        - Ensure good lighting
        - Keep gestures consistent but natural
        - Use clear, distinct movements
        """)
        
        # Clear training data
        if st.session_state.recognizer.training_data:
            if st.button("🗑️ Clear All Training Data", type="secondary"):
                st.session_state.recognizer.custom_actions = {}
                st.session_state.recognizer.training_data = []
                st.session_state.recognizer.model = None
                st.success("Training data cleared!")
                st.rerun()

def recognition_mode():
    st.header("Recognition Mode")
    
    if not st.session_state.recognizer.custom_actions:
        st.warning("No custom actions trained yet. Go to Training Mode to create your first actions.")
        return
    
    if st.session_state.recognizer.model is None:
        st.warning("Model not trained yet. Go to Training Mode and click 'Train Model'.")
        return
    
    # Settings
    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6, 0.1)
    with col_settings2:
        show_landmarks = st.checkbox("Show Hand Landmarks", True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Input")
        
        # Camera input
        camera_input = st.camera_input("Show your sign language gesture")
        
        if camera_input is not None:
            # Convert to OpenCV format
            bytes_data = camera_input.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Process image
            landmarks, results = st.session_state.recognizer.extract_landmarks(cv2_img)
            
            # Draw landmarks if enabled
            if show_landmarks and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        cv2_img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            # Display processed image
            st.image(cv2_img, channels="BGR", caption="Processed Image")
            
            # Make prediction
            if landmarks:
                predicted_action, confidence = st.session_state.recognizer.predict_action(landmarks)
                
                if predicted_action and confidence >= confidence_threshold:
                    st.success(f"**Detected Action: {predicted_action}**")
                    st.info(f"Confidence: {confidence:.2%}")
                    
                    # Add to history
                    st.session_state.prediction_history.append({
                        'prediction': predicted_action,
                        'confidence': confidence,
                        'timestamp': datetime.now()
                    })
                else:
                    if predicted_action:
                        st.warning(f"Low confidence prediction: {predicted_action} ({confidence:.2%})")
                    else:
                        st.warning("No confident prediction. Try adjusting your gesture.")
            else:
                st.warning("No hand detected. Make sure your hand is clearly visible.")
    
    with col2:
        st.subheader("Your Custom Actions")
        
        # Display trained actions
        for action, count in st.session_state.recognizer.custom_actions.items():
            st.write(f"📋 **{action}** ({count} samples)")
        
        st.markdown("---")
        
        # Prediction history
        if st.session_state.prediction_history:
            st.subheader("Recent Predictions")
            recent_predictions = st.session_state.prediction_history[-5:]
            
            for pred in reversed(recent_predictions):
                timestamp = pred['timestamp'].strftime("%H:%M:%S")
                st.write(f"**{pred['prediction']}** - {pred['confidence']:.1%} at {timestamp}")
            
            # Clear history button
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()

def model_management_mode():
    st.header("Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Model")
        st.write("Save your trained model and training data")
        
        if st.session_state.recognizer.training_data:
            if st.button("📤 Export Model Data"):
                export_data = st.session_state.recognizer.export_model()
                st.download_button(
                    label="⬇️ Download Model File",
                    data=export_data,
                    file_name=f"sign_language_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("No training data to export.")
    
    with col2:
        st.subheader("Import Model")
        st.write("Load a previously saved model")
        
        uploaded_file = st.file_uploader("Choose model file", type=['json'])
        
        if uploaded_file is not None:
            try:
                import_data = uploaded_file.read().decode('utf-8')
                success, message = st.session_state.recognizer.import_model(import_data)
                
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Failed to import: {str(e)}")
    
    st.markdown("---")
    
    # Model statistics
    st.subheader("Model Statistics")
    
    if st.session_state.recognizer.training_data:
        total_samples = len(st.session_state.recognizer.training_data)
        num_actions = len(st.session_state.recognizer.custom_actions)
        model_trained = st.session_state.recognizer.model is not None
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", total_samples)
        with col2:
            st.metric("Custom Actions", num_actions)
        with col3:
            st.metric("Model Status", "✅ Trained" if model_trained else "❌ Not Trained")
        
        # Training data distribution
        if st.session_state.recognizer.custom_actions:
            st.subheader("Training Data Distribution")
            df = pd.DataFrame(list(st.session_state.recognizer.custom_actions.items()), 
                            columns=['Action', 'Samples'])
            st.bar_chart(df.set_index('Action'))
    else:
        st.info("No model data available.")

if __name__ == "__main__":
    main()