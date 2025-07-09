#
# NaiveSAM toolkit v1.0
#

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import glob
import torch
import shutil
import pickle

import matplotlib.pyplot as plt
import ipywidgets as ipyw
import numpy as np

from ipywidgets import widgets, TwoByTwoLayout, GridspecLayout
from sam2.build_sam import build_sam2_video_predictor
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

import matplotlib.patheffects as patheffects


# NaiveSAM utility - nas #

alpha = 0.4
    
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='lime', marker='2', s=marker_size, linewidth=2.25)     #, edgecolors='white'
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='fuchsia', marker='1', s=marker_size, linewidth=2.25)  #, edgecolors='white'
 

def nas_show_points(coords, labels, ax, marker_size=200, obj_id=None, random_color=False):
    """ pts with color """
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='fuchsia', marker='1', s=marker_size, linewidth=2.25)  #, edgecolors='white'
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color=color, marker='2', s=marker_size, linewidth=2.25)

    

def show_mask(mask, ax, obj_id=None, random_color=False):
    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    
def nas_show_mask(mask, ax, obj_id=None, random_color=False, alpha=0.4):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    tmp = mask.reshape(h, w, 1).astype(np.uint8).copy()
    
    (cnts, _) = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tmp = cv2.drawContours(image=np.zeros_like(tmp), contours=cnts, contourIdx=-1, color=1, thickness=int(w/200)) #thickness=cv2.FILLED
    tmp = tmp.astype(bool)
    tmp_image = tmp * color.reshape(1, 1, -1)
    ax.imshow(tmp_image)
        
    if obj_id is not None and len(cnts) > 0:
        largest_cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(largest_cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) + int(w/50)
            cy = int(M["m01"] / M["m00"])
            ax.text(cx, cy, str(obj_id), color=color[:3], fontsize=int(w/50),
                    ha='center', va='center', weight='bold',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
                    
                    
def nas_draw_masklets_cv2(mask, image, obj_id=None, random_color=False, alpha=0.4):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab20")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], alpha])
        #print(f'color:{color}')
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    tmp = mask.reshape(h, w, 1).astype(np.uint8).copy()

    (cnts, _) = cv2.findContours(tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tmp = cv2.drawContours(image=image, contours=cnts, contourIdx=-1, color=(color[2]*255,color[1]*255,color[0]*255, alpha), thickness=int(w/300)) # mpl rgba to cv2 bgr
    

    # --- obj_id at center of mask ---
    if obj_id is not None and len(cnts) > 0:
        largest_cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(largest_cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"]) + int(w/50)
            cy = int(M["m01"] / M["m00"])
            
            # class border
            cv2.putText(
                tmp,
                str(obj_id),
                (cx, cy),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.7,        #font scale
                (0, 0, 0),  # black
                4,          #thickness
                lineType=cv2.LINE_AA
            )
            # class of object
            cv2.putText(
                tmp,
                str(obj_id),
                (cx, cy),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.7,
                (int(color[2]*255),int(color[1]*255),int(color[0]*255)),
                2,
                lineType=cv2.LINE_AA
            )

    return tmp


def nas_draw_and_save_masklets_cv2(video_dir, frame_names, vis_frame_stride, out_mask_path, video_segments):
    
    count=0
    for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):

        image = cv2.imread(os.path.join(video_dir, frame_names[out_frame_idx]), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # to BGRA
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (25, 25)
        # fontScale
        fontScale = 1.
        # Blue color in BGR
        color = (255, 255, 255)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.putText() method
        image = cv2.putText(image, f'frame {out_frame_idx}', org, font,
                           fontScale, color, thickness, cv2.LINE_AA)

        try :
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                image = nas_draw_masklets_cv2(out_mask, image, obj_id=out_obj_id)
            cv2.imwrite(f"./{out_mask_path}/{count:05}.jpg", image)
        except Exception as e:
            cv2.imwrite(f"./{out_mask_path}/{count:05}.jpg", image)
            print(f'missing mask: {e}')
            
        count+=1




# -- Convert SAM masklets to YOLO segmentation TXT formate  -- #

def nas_masklets_to_yolo_seg_txt(out_frame_idx, video_dir, frame_names, vis_frame_stride, video_segments):
    
    # write the yolo values to a text file
    with open(f"{video_dir}/{os.path.basename(frame_names[out_frame_idx]).split('.')[0]}.txt", "w") as f:
    
        # convert sam2 mask to countour coordinates
        try :
            
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():

                # [TODO: if mask >= pixels then go to work. ]
                if out_mask.any():

                    binary_mask = out_mask.squeeze().astype(np.uint8)

                    # Find the contours of the mask
                    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    largest_contour = max(contours, key=cv2.contourArea)

                    # smooth the mask
                    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
                    contour_approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    largest_contour = contour_approx

                    # Get the segmentation mask for object
                    segmentation = largest_contour.flatten().tolist()

                    # Write bounding coordinates of the object's segmentation mask to file in YOLO format
                    mask = segmentation

                    # load the image
                    image = cv2.imread(os.path.join(video_dir, frame_names[out_frame_idx]))
                    height, width = image.shape[:2]

                    # convert mask to numpy array of shape (N,2)
                    mask = np.array(mask).reshape(-1, 2)

                    # normalize the pixel coordinates
                    mask_norm = mask / np.array([width, height])

                    # reshape to 1-D list x y x y...
                    yolo = mask_norm.reshape(-1)

                    for val in yolo:
                        f.write("{} {:.6f}".format(out_obj_id,val))

                    f.write("\n")
                    
        except Exception as e:
            
                """https://github.com/ultralytics/ultralytics/issues/7981
                是的，預設情況下，Ultralytics YOLO模型在訓練期間會跳過沒有註解的圖像。這主要是為了確保訓練中使用的每張圖像都直接有助於學習，從而優化訓練過程的效率和效果。如果沒有物件的圖像（空白圖像）旨在有助於訓練（例如在沒有物件的地方教授模型），它仍然應該有一個關聯的註釋文件，只是沒有任何物件清單。如果您想要包含沒有註釋的圖像並確保它們不會被過濾掉，您可以考慮為每個此類圖像添加一個空註釋檔案。這樣，模型就可以理解這些是資料集的有意部分。
                Yes, by default, Ultralytics YOLO models skip images without annotations during training. This mainly ensures that every image used in training actively contributes to learning, thereby optimizing the efficiency and effectiveness of the process. If an image without objects (blank images) is meant to contribute to training, such as teaching the model where there are no objects, it should still have an annotation file, even if it contains no object listings. To include images without annotations and prevent them from being filtered out, you can add an empty annotation file for each one. This helps the model recognize that these images are an intentional part of the dataset.
                """
                print('Empty yolo-seg .txt', e)
                f.write("\n")


def nas_group_images_labels_to_yolo_data_dir(videos_list, folder_path, yolo_data):
    print(f'videos_list: {videos_list}')

    count = 0
    for video_name in tqdm(videos_list):

        video_path = os.path.join(folder_path, video_name)
        video_dir, ext = os.path.splitext(video_path)

        jpg_ls = sorted(os.listdir(f'{video_dir}'))
        for file in jpg_ls:
            if file.endswith(".txt"):
                # cp txt
                shutil.copy(f"{video_dir}/{file}", f"{yolo_data}/{video_name}_{file}")
                # cp jpg
                jpg = file.replace(".txt", ".jpg")
                shutil.copy(f"{video_dir}/{jpg}", f"{yolo_data}/{video_name}_{jpg}")
                count+=1

    print(f'* Total count: {count}')

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


def nas_create_yoylo_seg_dataset(yolo_data,):
    # Get 'TXT' list name of file
    txt_path_ls = [f"{yolo_data}/{txt}" for txt in sorted(os.listdir(f'{yolo_data}/')) if txt.endswith(".txt")]
    print(f"* txt_path_ls len: {len(txt_path_ls)}")


    # Get the train/val split, default 8:2 with seed '42'
    s_train, s_val = train_test_split(txt_path_ls, test_size=0.2, random_state=42)
    print(f"* train: {len(s_train)}, val: {len(s_val)}")
    print(f"* s_train[:3]:{s_train[:3]} \n* s_val[:3]:  {s_val[:3]}")


    # Moving images[train/val] and labels[train/val]
    def move(paths, folder):
        for p in paths:
            shutil.move(p, folder)

    # labels
    move(s_train, f"{yolo_data}/labels/train/")
    move(  s_val, f"{yolo_data}/labels/val/")

    # images
    jpg_train = [jpg.replace(".txt", ".jpg") for jpg in s_train]
    jpg_val   = [jpg.replace(".txt", ".jpg") for jpg in s_val  ]
    print(f"* jpg_train: {len(jpg_train)}, val: {len(jpg_val)}")
    print(f"* jpg_train[:3]:{jpg_train[:3]} \n* s_val[:3]:  {jpg_val[:3]}")

    move(jpg_train, f"{yolo_data}/images/train/")
    move(  jpg_val, f"{yolo_data}/images/val/")





# NaiveSAM UI #

class NaiveSamTool:
    """
    A tool for interactive image annotation in a Jupyter Notebook environment.
    It integrates Matplotlib display, ipywidgets controls, and a hypothetical SAM2 predictor.
    """

    def __init__(self,
                 predictor_instance,        # SAM predictor instance
                 inference_state_instance,  # Inference state instance
                 show_points_func,          # Helper function to display points (from external module)
                 show_mask_func,            # Helper function to display masks (from external module)
                 frame_count: int,          # Total number of frames
                 video_dir: str,            # Directory storing frame images
                 frame_names: list[str],    # List of frame image filenames
                 folder_path: str,          # Root folder for saving .pts files (original: folder_path)
                 video_name_for_pts: str,   # Video name for .pts file naming (original: video_name)
                 # --- Pass all externally defined ipywidgets and Matplotlib objects ---
                 class_list_widget: widgets.RadioButtons,
                 ng_widget: widgets.Checkbox,
                 int_range_widget: widgets.IntSlider,
                 frame_count_output_widget: ipyw.Output,
                 pts_list_out_widget: ipyw.Output,
                 pts_save_button_widget: widgets.Button,
                 pts_save_button_output_widget: ipyw.Output,
                 pts_clear_button_widget: widgets.Button,
                 pts_clear_button_output_widget: ipyw.Output,
                 figure_instance: plt.Figure,
                 axes_instance: plt.Axes
                ):
                
        # --- External dependencies and initialization data ---
        self.predictor = predictor_instance
        self.inference_state = inference_state_instance
        self.show_points = show_points_func
        self.show_mask = show_mask_func
        self.frame_count = frame_count
        self.video_dir = video_dir
        self.frame_names = frame_names
        self.folder_path = folder_path  # For pts saving
        self.video_name_for_pts = video_name_for_pts # For pts file naming

        # --- Matplotlib figure objects ---
        self.fig = figure_instance
        self.ax = axes_instance

        # --- ipywidgets objects ---
        self.class_list = class_list_widget
        self.ng = ng_widget
        self.int_range = int_range_widget
        self.frame_count_output = frame_count_output_widget
        self.pts_list_out = pts_list_out_widget
        self.pts_save_button = pts_save_button_widget
        self.pts_save_button_output = pts_save_button_output_widget
        self.pts_clear_button = pts_clear_button_widget
        self.pts_clear_button_output = pts_clear_button_output_widget

        # --- Internal state variables ---
        self.prompts = {}  # hold all the clicks we add for visualization for predictor
        self.obj_pts_list = []  # Independent list to record all click info for the current frame
        self.out_frame_idx = 0  # Current displayed frame index

        # --- Initialization and event connections ---
        self._setup_initial_display()
        self._connect_events()
        self._setup_layout()

        # Reset predictor state on each startup
        self.predictor.reset_state(self.inference_state)


    def _setup_initial_display(self):
        """Sets up the initial display state of the Matplotlib figure."""
        # Make the image fill the entire subplot area (called from fig instance)
        self.fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0)

        # Turn off axes
        self.ax.axis('off')

        # Default: load and display frame 0 image
        self.ax.set_title(f'Frame: {self.out_frame_idx}')
        self.ax.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[self.out_frame_idx])))
        self.fig.canvas.draw_idle() # Ensure initial display


    def _connect_events(self):
        """Connects ipywidgets and Matplotlib events to their respective handlers."""
        self.int_range.observe(self._on_value_change, names='value')
        self.pts_save_button.on_click(self._on_pts_save_button_clicked)
        self.pts_clear_button.on_click(self._on_pts_clear_button_clicked)
        self.fig.canvas.mpl_connect('button_press_event', self._onclick) # Matplotlib click event


    def _setup_layout(self):
        """Sets up the ipywidgets grid layout."""
        self.grid = GridspecLayout(16, 4, height='520px', width='940px', border='2px solid red') # Assuming this GridspecLayout is defined externally

        # top-bar : frame numbers + point save button + clear button
        self.grid[0, 0] = self.int_range
        self.grid[0, 1] = self.pts_save_button
        self.grid[0, 2] = self.pts_clear_button

        # 2nd row: n-g setting, pts displayer
        # 3rd row: Artifacts labels
        #self.grid[1, 0] = self.ng # positive/negative Checkbox
        #self.grid[2:10, 0] = self.class_list # Object Labels RadioButtons :1 (0)
        self.grid[1, 0] = ipyw.HBox([self.ng], layout=ipyw.Layout(justify_content='center'))
        self.grid[2:, 0] = ipyw.HBox([self.class_list], layout=ipyw.Layout(justify_content='center'))
        self.grid[2:10, 1:3] = self.pts_list_out # Output area for points list 1:3 (1-2)

    def _on_value_change(self, change):
        """Handles frame slider value change event."""
        with self.frame_count_output:
            self.frame_count_output.clear_output(wait=True) # Use wait=True to prevent flickering
            print( 'frame_count_output', change['new'])

        self.out_frame_idx = int(change['new'])

        # Clear all drawings and reload image every time frame changes
        self.ax.clear()
        self.ax.set_title(f'Frame: {self.out_frame_idx}')
        self.ax.axis('off')
        self.ax.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[self.out_frame_idx])))
        self.fig.canvas.draw_idle() # Ensure figure update

        # Reset current frame's click list and predictor state
        self.obj_pts_list = []
        self.prompts = {}
        self.predictor.reset_state(self.inference_state)


    def _on_pts_save_button_clicked(self, b):
        """Handles the save points data button event."""
        self.pts_save_button_output.clear_output(wait=True)

        with self.pts_save_button_output:
            print("Button clicked. Saving pts with frame number.")

            # Ensure save directory exists
            # Filename example: ./robot_example/videos/20250602-2_720x480_end_f_00000.pts
            pts_filename = os.path.join(self.folder_path, f"{self.video_name_for_pts}_f_{self.out_frame_idx}.pts")
            os.makedirs(os.path.dirname(pts_filename), exist_ok=True)

            try:
                with open(pts_filename, "wb") as fp:
                    pickle.dump(self.obj_pts_list, fp) # Use pickle module
                print(f"Saved points for frame {self.out_frame_idx} to {pts_filename}")
                self.obj_pts_list = [] # Clear current frame's click list, prepare for next frame
                self.prompts = {} # Clear prompts
                self.predictor.reset_state(self.inference_state) # Reset predictor state

                self.pts_list_out.clear_output() # Clear click list output area
                
                # Redraw image, clear points and mask for current frame
                self.ax.clear()
                self.ax.set_title(f'Frame: {self.out_frame_idx}')
                self.ax.axis('off')
                self.ax.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[self.out_frame_idx])))
                self.fig.canvas.draw_idle()

            except Exception as e:
                print(f"Error saving points: {e}")


    def _on_pts_clear_button_clicked(self, b):
        """Handles the clear points data button event."""
        self.obj_pts_list = []  # Clear click data list
        self.prompts = {} # Clear predictor's prompts dictionary
        self.predictor.reset_state(self.inference_state) # Reset predictor's internal state

        self.pts_clear_button_output.clear_output(wait=True)
        with self.pts_clear_button_output:
            print("Button clicked. Cleared points and reset state for current frame.")

        self.pts_list_out.clear_output() # Clear click list output area

        # Clear Matplotlib figure's drawing content, and reload current frame image
        self.ax.clear()
        self.ax.set_title(f'Frame: {self.out_frame_idx}')
        self.ax.axis('off')
        self.ax.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[self.out_frame_idx])))
        self.fig.canvas.draw_idle() # Force Matplotlib redraw


    @widgets.Output().capture(clear_output=True, wait=True)
    # @self.pts_list_out.capture(wait=True)
    def _onclick(self, event):
        """
        Handles mouse click events on the Matplotlib figure.
        """
        if event.xdata is None or event.ydata is None:
            with self.pts_list_out:
                print("Clicked outside image area.")
            return

        with self.pts_list_out: # Capture all print statements to self.pts_list_out widget
            # Get object ID and label value
            # class_list.value will be a string like "0 robot", take the first two characters and convert to int
            obj_number_str = self.class_list.value[:2].strip()
            
            # ng.value will be True or False, directly convert to int (True -> 1, False -> 0)
            ng_label_val = int(self.ng.value)

            # Input validation
            try:
                obj_number = int(obj_number_str)
                if not (0 <= obj_number <= 99): # Assume ID range 0-99
                    print("Object ID must be between 0 and 99.")
                    return
            except ValueError:
                print("Invalid Object ID. Please select from the labels.")
                return

            if ng_label_val not in [0, 1]:
                print("Label must be 0 (Negative) or 1 (Positive).")
                return

            point_data = [self.out_frame_idx, obj_number, ng_label_val, int(event.xdata), int(event.ydata)]
            self.obj_pts_list.append(point_data)
            print(f"{point_data}")

            # After each click, redraw the current frame, then display all points and masks
            self.ax.clear()
            self.ax.set_title(f'Frame: {self.out_frame_idx}')
            self.ax.axis('off')
            self.ax.imshow(Image.open(os.path.join(self.video_dir, self.frame_names[self.out_frame_idx])))

            # Display all click points for the current frame (only draw points for current frame)
            current_frame_clicks = [p for p in self.obj_pts_list if p[0] == self.out_frame_idx]
            for frn, obn, lab, x, y in current_frame_clicks:
                points_arr = np.array([[x, y]], dtype=np.float32)
                labels_arr = np.array([lab], np.int32)
                self.show_points(points_arr, labels_arr, self.ax, obj_id=obn)

            # Perform real-time mask prediction
            # Collect all clicks for the current object in the current frame
            obj_specific_points = np.array([p[3:5] for p in current_frame_clicks if p[1] == obj_number], dtype=np.float32)
            obj_specific_labels = np.array([p[2] for p in current_frame_clicks if p[1] == obj_number], np.int32)


            if obj_specific_points.size > 0:
                self.prompts[obj_number] = obj_specific_points, obj_specific_labels
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                    inference_state=self.inference_state,
                    frame_idx=self.out_frame_idx,
                    obj_id=obj_number,
                    points=obj_specific_points,
                    labels=obj_specific_labels,
                )

                # Display mask (only contour, to avoid transparency stacking issues)
                for i, out_obj_id in enumerate(out_obj_ids):
                    # Assuming out_mask_logits[i] is a single mask logit
                    self.show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), self.ax, obj_id=out_obj_id)
            self.fig.canvas.draw_idle() # Force redraw update

    def display_app(self):
        """Displays the application's UI in the Jupyter Notebook."""
        display(self.grid)
        
        
    #
    # merge all vids_frame#.pts to video_name.pts and # Add new points by "obj_pts_list"
    #
    
    
    def merge_pts(self):
        """Merges all individual frame .pts files into a single video .pts file."""
        pts_temp = []

        pattern = os.path.join(f"{self.folder_path}", f"{self.video_name_for_pts}_f_*.pts")
        matching_files = sorted(glob.glob(pattern))
        matching_files.sort(key=lambda p: int(p.split("f_")[1].split(".")[0]))  # Sort by frame number

        for pt_file in matching_files:

            with open(pt_file, "rb") as fp:
                for pts_list_temp in pickle.load(fp):
                    pts_temp.append(pts_list_temp)

        #print(type(pts_temp), pts_temp)

        # Save merged pts
        with open(f"{self.folder_path}{self.video_name_for_pts}.pts","wb") as fp:
            pickle.dump(pts_temp, fp)


        def show_pts_temp():
            """Helper function to print loaded points for debugging."""
            print(f"Add pts to predictor:")
            for pts in pts_temp:
                print(f"\t{pts}")

        show_pts_temp()
        
    def add_new_pts_to_predictor(self):
        """
        Loads points from the merged .pts file and adds them to the predictor.

        Note! The read and write paths for pts files must be consistent!
        """

        prompts = {}  # hold all the clicks we add for visualization

        folder_path=self.folder_path
        video_name=self.video_name_for_pts
        predictor=self.predictor

        obj_pts_list = []
        
        # Load pts
        with open(f"{folder_path}{video_name}.pts","rb") as fp:
            pk_list = pickle.load(fp)
            #print(type(pk_list), pk_list)
        obj_pts_list = pk_list

        # Add pts
        for frn, obn, lab, x, y in obj_pts_list:
            ann_frame_idx = frn  # the frame index we interact with
            ann_obj_id = obn  # give a unique id to each object we interact with (it can be any integers)

            points =  np.array([[x, y]], dtype=np.float32)
            labels = np.array([lab], np.int32)

            prompts[ann_obj_id] = points, labels
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=self.inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

