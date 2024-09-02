import azure.cognitiveservices.speech as speechsdk
import time
import os


def RecognizeTextFromAudioFile(audioFilePath):
    """Performs Speech recognition on an audio file and returns the recognized text"""
    speech_key = str(os.getenv("AZURE_SPEECH_KEY"))
    service_region = str(os.getenv("AZURE_SPEECH_REGION"))
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_recognition_language = "nl-NL"  # Set language to Dutch
    
    audio_config = speechsdk.audio.AudioConfig(filename=audioFilePath)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False

    def stop_cb(evt: speechsdk.SessionEventArgs):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    # List to hold the transcriptions
    transcription_list = []

    # Callback function to handle recognized text
    def recognized(evt):
        print(f"Recognized: {evt.result.text}")
        nonlocal transcription_list
        transcription_list.append(evt.result.text)

    speech_recognizer.recognized.connect(recognized)
    # Stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    print("Starting speech recognition on {}...", audioFilePath)
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()
    
    return " ".join(transcription_list)