def _extract_audio(video_path, audio_path, job_id_log_prefix=""):
    logger.info(f"{job_id_log_prefix} Extracting audio from {video_path} to {audio_path}")
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')
    video.close()
    logger.info(f"{job_id_log_prefix} Audio extracted.")

def _audio_to_text(wav_path, srt_path, job_id_log_prefix=""):
    logger.info(f"{job_id_log_prefix} Transcribing {wav_path} to {srt_path} using {device}")
    
    try:
        # Load model if not already loaded
        model = load_whisper_model()
        
        # Transcribe with GPU acceleration
        start_time = time.time()
        
        # Set FP16 based on device - GPU can use FP16 for faster processing
        use_fp16 = device == "cuda"
        
        result = model.transcribe(
            wav_path, 
            fp16=use_fp16,
            verbose=False  # Reduce logging noise
        )
        
        processing_time = time.time() - start_time
        logger.info(f"{job_id_log_prefix} Transcription completed in {processing_time:.2f}s using {device}")
        
        # Write SRT file
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result["segments"]):
                start, end, text = segment['start'], segment['end'], segment['text'].strip()
                f.write(f"{i+1}\n{format_whisper_timestamp(start)} --> {format_whisper_timestamp(end)}\n{text}\n\n")
        
        logger.info(f"{job_id_log_prefix} Transcription complete. SRT saved to {srt_path}")
        
        # Clear GPU cache after transcription
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Error during transcription: {str(e)}", exc_info=True)
        raise

def _translate_srt(original_srt_path, target_lang, translated_srt_path, job_id_log_prefix=""):
    if target_lang.lower() in ['en', 'english']:
        logger.info(f"{job_id_log_prefix} Target is English, skipping translation.")
        shutil.copyfile(original_srt_path, translated_srt_path)
        return
    logger.info(f"{job_id_log_prefix} Translating {original_srt_path} to {target_lang}")
    subs = SubRipFile.open(original_srt_path, encoding='utf-8')
    translator = GoogleTranslator(source='auto', target=target_lang)
    for sub in subs:
        try:
            translated = translator.translate(sub.text)
            sub.text = translated if translated else sub.text
        except Exception as e_trans:
            logger.warning(f"{job_id_log_prefix} Translation failed for '{sub.text[:30]}...': {e_trans}")
    subs.save(translated_srt_path, encoding='utf-8')
    logger.info(f"{job_id_log_prefix} Translation complete.")
    
def _add_captions_to_video(video_path, srt_path, output_path, font_opts, job_id_log_prefix=""):
    """Add captions to video with enhanced error handling and text rendering."""
    try:
        logger.info(f"{job_id_log_prefix} Adding captions from {srt_path} to {video_path}")
        
        # Validate input files
        if not all(os.path.exists(f) for f in [video_path, srt_path]):
            raise FileNotFoundError("Input video or SRT file not found")
        
        # Load video and subtitles
        video = VideoFileClip(video_path)
        subs = SubRipFile.open(srt_path, encoding='utf-8')
        
        # Process font options with robust fallback
        font_name = font_opts.get('family', 'Arial.ttf')
        font_size = font_opts.get('size', 24)
        font_color = font_opts.get('color', '#FFFFFF')
        
        # Font loading with multiple fallbacks
        font = None
        font_paths_to_try = [
            os.path.join(FONT_FOLDER, font_name),
            font_name,  # Try system font
            os.path.join(FONT_FOLDER, "Arial.ttf"),
            "arial.ttf",  # Common system fallback
            os.path.join(FONT_FOLDER, "LiberationSans-Regular.ttf")
        ]
        
        for fp in font_paths_to_try:
            try:
                font = ImageFont.truetype(fp, font_size)
                logger.info(f"{job_id_log_prefix} Using font: {fp}")
                break
            except (IOError, OSError):
                continue
                
        if font is None:
            raise RuntimeError(f"Could not load any fallback font from: {font_paths_to_try}")

        def text_overlay_func(get_frame, t):
            """Process each frame to add captions."""
            try:
                frame_array = get_frame(t)
                img = Image.fromarray(frame_array)
                draw = ImageDraw.Draw(img)
                
                # Get active subtitles for current time
                active_texts = []
                for sub in subs:
                    start = sub.start.ordinal / 1000.0
                    end = sub.end.ordinal / 1000.0
                    if start <= t <= end:
                        text = sub.text
                        # Handle RTL languages like Arabic
                        if any('\u0600' <= char <= '\u06FF' for char in text):
                            text = get_display(arabic_reshaper.reshape(text))
                        active_texts.append(text)
                
                if not active_texts:
                    return np.array(img)
                
                full_caption = " ".join(active_texts)
                
                # Improved text wrapping
                max_width = img.width * 0.9
                lines = []
                words = full_caption.split()
                current_line = ""
                
                for word in words:
                    test_line = f"{current_line} {word}" if current_line else word
                    text_width = draw.textlength(test_line, font=font)
                    
                    if text_width <= max_width:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Calculate text position
                line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + 5
                total_text_height = len(lines) * line_height
                margin = img.height * 0.05
                base_y = img.height - total_text_height - margin
                
                # Draw each line with stroke/border
                stroke_width = 2
                stroke_color = (0, 0, 0)  # Black stroke
                text_color = tuple(int(font_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))  # Hex to RGB
                
                for i, line in enumerate(lines):
                    if not line:
                        continue
                        
                    text_width = draw.textlength(line, font=font)
                    x = (img.width - text_width) / 2
                    y = base_y + (i * line_height)
                    
                    # Draw stroke (outline)
                    for dx in [-stroke_width, stroke_width]:
                        for dy in [-stroke_width, stroke_width]:
                            draw.text((x + dx, y + dy), line, font=font, fill=stroke_color)
                    
                    # Draw main text
                    draw.text((x, y), line, font=font, fill=text_color)
                
                return np.array(img)
            
            except Exception as e:
                logger.error(f"{job_id_log_prefix} Error processing frame at {t}s: {str(e)}")
                return get_frame(t)  # Return original frame on error
        
        # Process video with progress tracking
        logger.info(f"{job_id_log_prefix} Starting video processing...")
        captioned_clip = video.fl(text_overlay_func, apply_to=['video'])
        
        # Write output with optimized settings
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        captioned_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset='medium',
            ffmpeg_params=["-crf", "23"],
            logger='bar' if logger.level <= logging.INFO else None
        )
        
        logger.info(f"{job_id_log_prefix} Successfully created captioned video: {output_path}")
        
    except Exception as e:
        logger.error(f"{job_id_log_prefix} Error in _add_captions_to_video: {str(e)}", exc_info=True)
        raise
    finally:
        # Ensure resources are cleaned up
        if 'video' in locals():
            video.close()
        if 'captioned_clip' in locals():
            captioned_clip.close()
