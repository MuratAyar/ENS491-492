import dearpygui.dearpygui as dpg
import asyncio
from agents.orchestration.orchestrator import Orchestrator

# Initialize UI
dpg.create_context()

# Initialize Orchestrator
orchestrator = Orchestrator()

# Define Colors
dark_gray = (25, 25, 25)
light_gray = (220, 220, 220)
accent_color = (75, 140, 255)
text_black = (0, 0, 0)

# Define Assets
background_image = "24049.jpg"
logo_image = "logo1.jpg"

# Function to Resize UI Elements on Window Resize
def resize_callback(sender, app_data):
    width, height = dpg.get_viewport_width(), dpg.get_viewport_height()
    dpg.configure_item("home_page", width=width, height=height)
    dpg.configure_item("bg_texture", width=width, height=height)
    dpg.configure_item("logo_texture", pos=(width - 120, 20))
    dpg.configure_item("title_text", pos=(width // 2 - 300, height // 3 - 50))
    dpg.configure_item("description_text", pos=(width // 2 - 300, height // 3))
    dpg.configure_item("analyze_button", pos=(width // 2 - 150, height // 3 + 80))

# Function to Analyze Transcript
def analyze_callback(sender, app_data):
    transcript = dpg.get_value("transcript_input")
    if transcript:
        dpg.set_value("result_text", "Analyzing... Please wait.")
        
        async def run_analysis():
            result = await orchestrator.process_transcript(transcript)
            if "error" in result:
                dpg.set_value("result_text", f"Error: {result['error']}")
            else:
                output_text = (
                    f"Sentiment: {result.get('sentiment', 'N/A')}\n"
                    f"Tone: {result.get('tone', 'N/A')}\n"
                    f"Empathy Level: {result.get('empathy', 'N/A')}\n"
                    f"Responsiveness: {result.get('responsiveness', 'N/A')}\n"
                    f"Primary Category: {result.get('primary_category', 'N/A')}\n"
                    f"Score: {result.get('caregiver_score', 'N/A')}/5"
                )
                dpg.set_value("result_text", output_text)
        
        asyncio.run(run_analysis())
    else:
        dpg.set_value("result_text", "Please enter a transcript!")

# Create Viewport
dpg.create_viewport(title='Caregiver Monitoring System', width=1280, height=800, resizable=True, decorated=False)
dpg.setup_dearpygui()

# Load Background Image
with dpg.texture_registry():
    width, height, channels, data = dpg.load_image(background_image)
    dpg.add_static_texture(width, height, data, tag="bg_texture")

with dpg.texture_registry():
    logo_width, logo_height, logo_channels, logo_data = dpg.load_image(logo_image)
    dpg.add_static_texture(logo_width, logo_height, logo_data, tag="logo_texture")

# Font Scaling System
with dpg.font_registry():
    large_font = dpg.add_font("C:\\Windows\\Fonts\\arial.ttf", 50)
    medium_font = dpg.add_font("C:\\Windows\\Fonts\\arial.ttf", 30)

# Create Home Page
with dpg.window(tag="home_page", width=1280, height=800, no_resize=True, no_title_bar=True):
    dpg.add_image("bg_texture")
    
    # Logo
    dpg.add_image("logo_texture", width=100, height=100, pos=(20, 20))

    # Title & Description
    dpg.add_text("Caregiver Monitoring & Evaluation System", pos=(400, 250),
                 color=text_black, tag="title_text", wrap=600)
    dpg.bind_font(large_font)  # Büyük başlık fontunu uygula
    
    dpg.add_text("This tool evaluates caregiver-child interactions using AI.", pos=(400, 300),
                 color=text_black, tag="description_text", wrap=600)
    dpg.bind_font(medium_font)  # Açıklama yazısına orta boy fontu uygula

    # Button
    dpg.add_button(label="Analyze Caregiver", callback=lambda: dpg.show_item("analysis_window"),
                   pos=(540, 380), width=300, height=60, tag="analyze_button")

# Create Analysis Page
with dpg.window(label="Analysis Page", tag="analysis_window", show=False, width=800, height=600,
                pos=(240, 100), no_title_bar=True, no_resize=True):
    dpg.add_button(label="Return to Home Page", callback=lambda: dpg.hide_item("analysis_window"),
                   width=200, height=40)
    dpg.add_text("Enter a transcript or upload a file", color=accent_color, bullet=True)
    dpg.add_input_text(multiline=True, width=700, height=250, tag="transcript_input", hint="Paste the conversation here...")
    dpg.add_button(label="Start Analysis", callback=analyze_callback, width=300, height=60)
    dpg.add_text("", tag="result_text", wrap=700, color=light_gray)

# Set Primary Window
dpg.set_primary_window("home_page", True)

# Set Resize Callback for Dynamic UI Scaling
dpg.set_viewport_resize_callback(resize_callback)

# Run UI
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
