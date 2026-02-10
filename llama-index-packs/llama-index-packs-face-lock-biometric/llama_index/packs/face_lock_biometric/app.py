"""
Face Lock: Streamlit Web Interface.

Provides a web UI for both synthetic and real-face workflows with:
- Synthetic character generation (BIPA-safe)
- Real face analysis (BIPA-regulated, consent required)
- Prompt generation for Reve/Flux/Midjourney
- Drift detection across generated images
- Analog film post-processing
"""

from typing import Optional


def main() -> None:
    """Launch the Face Lock Streamlit web application."""
    try:
        import streamlit as st
    except ImportError:
        raise ImportError(
            "Streamlit is required for the Face Lock web interface. "
            "Install with: pip install streamlit"
        )

    import json
    import tempfile
    from pathlib import Path

    import numpy as np

    from .analog_film import AnalogFilmProcessor, FilmPreset, FILM_PRESETS
    from .bipa_consent import ConsentStatus
    from .face_lock import FaceAnalyzer, FacialMeasurements
    from .prompt_generators import Platform, PromptGenerator
    from .workflows import (
        CharacterProfile,
        RealFaceWorkflow,
        SyntheticWorkflow,
        WorkflowMode,
    )

    st.set_page_config(
        page_title="Face Lock — Biometric Character Consistency",
        page_icon="🔒",
        layout="wide",
    )

    st.title("Face Lock")
    st.caption("Biometric Character Consistency for AI Image Generation")

    # Sidebar — mode selection
    st.sidebar.header("Workflow Mode")
    mode = st.sidebar.radio(
        "Select workflow:",
        options=["Synthetic (BIPA-Safe)", "Real Face (BIPA-Regulated)"],
        index=0,
        help=(
            "Synthetic: Generate random characters — no real photos, no legal "
            "risk. Real Face: Analyze uploaded photos — requires BIPA consent."
        ),
    )

    st.sidebar.divider()
    st.sidebar.header("Platform")
    platform_name = st.sidebar.selectbox(
        "Target platform:",
        options=["Reve", "Flux", "Midjourney", "Generic"],
    )
    platform_map = {
        "Reve": Platform.REVE,
        "Flux": Platform.FLUX,
        "Midjourney": Platform.MIDJOURNEY,
        "Generic": Platform.GENERIC,
    }
    platform = platform_map[platform_name]

    include_technical = st.sidebar.checkbox(
        "Include technical values in prompts", value=False
    )

    # Initialize session state
    if "measurements" not in st.session_state:
        st.session_state.measurements = None
    if "character_name" not in st.session_state:
        st.session_state.character_name = ""

    # ======== SYNTHETIC WORKFLOW ========
    if mode == "Synthetic (BIPA-Safe)":
        st.header("Synthetic Character Generation")
        st.info(
            "Generate characters from scratch with random but biologically "
            "plausible measurements. No photo uploads required — zero legal risk."
        )

        col1, col2 = st.columns(2)

        with col1:
            char_name = st.text_input(
                "Character name:", value="Character_001"
            )
            seed = st.number_input(
                "Random seed (for reproducibility):",
                min_value=0,
                max_value=999999,
                value=42,
            )

        with col2:
            st.subheader("Override measurements")
            override_jaw = st.slider(
                "Gonial angle (°)", 118.0, 140.0, 128.0, 0.5
            )
            override_index = st.slider(
                "Facial index", 1.15, 1.50, 1.30, 0.01
            )
            override_canthal = st.slider(
                "Canthal tilt (°)", -6.0, 8.0, 2.0, 0.5
            )
            override_nasal = st.slider(
                "Nasal tip rotation (°)", 85.0, 115.0, 97.0, 0.5
            )

        if st.button("Generate Character", type="primary"):
            workflow = SyntheticWorkflow()
            profile = workflow.create_character(
                name=char_name,
                seed=int(seed),
                overrides={
                    "gonial_angle": override_jaw,
                    "facial_index": override_index,
                    "canthal_tilt": override_canthal,
                    "nasal_tip_rotation": override_nasal,
                },
            )
            st.session_state.measurements = profile.measurements
            st.session_state.character_name = char_name
            st.success(f"Character '{char_name}' generated!")

    # ======== REAL FACE WORKFLOW ========
    else:
        st.header("Real Face Analysis")
        st.warning(
            "⚠️ This workflow processes real biometric data. Illinois BIPA "
            "applies ($1,000-$5,000 per violation). Consent is required "
            "before any analysis."
        )

        # Consent form
        with st.expander("BIPA Consent Form", expanded=True):
            st.markdown(
                """
                **Notice of Biometric Data Collection**

                By proceeding, you acknowledge that:
                1. Facial geometry measurements will be extracted from your uploaded image
                2. Data will be retained for a maximum of **30 days**
                3. Data will **not** be sold or shared with third parties
                4. Purpose: AI character consistency generation
                5. You may revoke consent at any time
                """
            )
            subject_id = st.text_input(
                "Subject identifier (email or name):"
            )
            consent_given = st.checkbox(
                "I have read and agree to the above terms"
            )

        uploaded = st.file_uploader(
            "Upload reference photo:",
            type=["jpg", "jpeg", "png", "webp"],
        )

        char_name = st.text_input("Character name:", value="Character_001")

        if st.button("Analyze Face", type="primary"):
            if not consent_given or not subject_id:
                st.error(
                    "BIPA consent is required. Please provide your identifier "
                    "and agree to the terms."
                )
            elif uploaded is None:
                st.error("Please upload a reference photo.")
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    workflow = RealFaceWorkflow(
                        consent_storage_dir=tmpdir,
                        collector_entity="FaceLock Web App",
                    )
                    workflow.request_consent(subject_id)

                    # Save uploaded file temporarily
                    import cv2

                    file_bytes = np.frombuffer(
                        uploaded.read(), dtype=np.uint8
                    )
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if image is None:
                        st.error("Could not decode the uploaded image.")
                    else:
                        try:
                            profile = workflow.analyze_face(
                                image_input=image,
                                subject_identifier=subject_id,
                                character_name=char_name,
                            )
                            st.session_state.measurements = (
                                profile.measurements
                            )
                            st.session_state.character_name = char_name
                            st.success(
                                f"Face analyzed! Character '{char_name}' created."
                            )
                        except ValueError as e:
                            st.error(f"Analysis failed: {e}")
                        finally:
                            workflow.close()

    # ======== RESULTS DISPLAY ========
    if st.session_state.measurements is not None:
        m = st.session_state.measurements
        st.divider()

        col_meas, col_prompt = st.columns(2)

        with col_meas:
            st.subheader("Locked Measurements")
            data = m.to_dict()
            for key, value in data.items():
                unit = ""
                if key in ("gonial_angle", "canthal_tilt", "nasal_tip_rotation"):
                    unit = "°"
                st.metric(label=key.replace("_", " ").title(), value=f"{value}{unit}")

        with col_prompt:
            st.subheader(f"Prompt ({platform_name})")
            gen = PromptGenerator(
                platform=platform, include_technical=include_technical
            )
            context = st.text_input(
                "Additional context (scene, clothing, etc.):", value=""
            )
            positive, negative = gen.generate(m, additional_context=context)

            st.text_area(
                "Positive prompt:", value=positive, height=150, key="pos"
            )
            st.text_area(
                "Negative prompt:", value=negative, height=100, key="neg"
            )

            # Export as JSON
            st.download_button(
                "Download Character Profile (JSON)",
                data=json.dumps(
                    {
                        "name": st.session_state.character_name,
                        "measurements": m.to_dict(),
                        "prompts": {
                            "platform": platform_name,
                            "positive": positive,
                            "negative": negative,
                        },
                    },
                    indent=2,
                ),
                file_name=f"{st.session_state.character_name}_profile.json",
                mime="application/json",
            )

        # ======== FILM PROCESSING SECTION ========
        st.divider()
        st.subheader("Analog Film Processing (Optional)")

        film_preset_name = st.selectbox(
            "Film stock preset:",
            options=[
                "None",
                "Agfacolor CNS2 Expired (1972)",
                "Kodachrome 25",
                "Kodak Vision3 500T (Pushed +2)",
            ],
        )

        preset_map = {
            "Agfacolor CNS2 Expired (1972)": FilmPreset.AGFA_CNS2_EXPIRED_1972,
            "Kodachrome 25": FilmPreset.KODACHROME_25,
            "Kodak Vision3 500T (Pushed +2)": FilmPreset.KODAK_VISION3_500T_PUSH2,
        }

        if film_preset_name != "None":
            film_uploaded = st.file_uploader(
                "Upload generated image for film processing:",
                type=["jpg", "jpeg", "png", "webp"],
                key="film_upload",
            )

            if film_uploaded and st.button("Apply Film Effect"):
                import cv2

                file_bytes = np.frombuffer(
                    film_uploaded.read(), dtype=np.uint8
                )
                film_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if film_image is not None:
                    processor = AnalogFilmProcessor(
                        preset=preset_map[film_preset_name]
                    )
                    processed = processor.process(film_image)

                    # Display side by side
                    col_orig, col_film = st.columns(2)
                    with col_orig:
                        st.image(
                            cv2.cvtColor(film_image, cv2.COLOR_BGR2RGB),
                            caption="Original",
                            use_container_width=True,
                        )
                    with col_film:
                        st.image(
                            cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
                            caption=f"Processed ({film_preset_name})",
                            use_container_width=True,
                        )

        # ======== DRIFT DETECTION SECTION ========
        st.divider()
        st.subheader("Drift Detection")
        st.caption(
            "Upload generated images to check if they match the locked "
            "biometric parameters."
        )

        drift_files = st.file_uploader(
            "Upload generated images for drift check:",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="drift_upload",
        )

        if drift_files and st.button("Run Drift Detection"):
            import cv2

            from .face_lock import DriftDetector

            analyzer = FaceAnalyzer()
            detector = DriftDetector(target=m, analyzer=analyzer)

            for i, f in enumerate(drift_files):
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    st.warning(f"Could not decode {f.name}")
                    continue

                try:
                    report = detector.check(img, image_index=i)
                    if report.is_within_tolerance:
                        st.success(f"✅ {f.name}: All measurements within tolerance")
                    else:
                        st.warning(f"⚠️ {f.name}: Drift detected")
                        for detail in report.details:
                            st.text(f"  → {detail}")
                except ValueError as e:
                    st.error(f"❌ {f.name}: {e}")

            analyzer.close()

    # Footer
    st.divider()
    st.caption(
        "Face Lock — Parametric character consistency for AI image generation. "
        "Synthetic characters are BIPA-safe. Real face analysis requires consent."
    )


if __name__ == "__main__":
    main()
