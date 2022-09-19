<script lang="ts">
  import type {GraphModel} from "@tensorflow/tfjs";
  import * as tf from "@tensorflow/tfjs";
  import {onMount} from "svelte";
  import srcImage from "./assets/sample.jpg";

  let elVideo: HTMLVideoElement;
  let elCanvas: HTMLCanvasElement;
  let context: CanvasRenderingContext2D;
  let model: GraphModel;
  let stream: MediaStream;
  let elImage: HTMLImageElement;

  enum Keys {
    num_detection,
    detection_boxes,
    detection_classes,
    detection_scores,
    detection_keypoints,
    detection_keypoints_scores,
  }

  async function init(): Promise<void> {
    model = await tf.loadGraphModel("./web_model/model.json");

    // elVideo.srcObject = await navigator.mediaDevices.getUserMedia({
    //   video: {
    //     facingMode: "environment",
    //   }
    // });
  }

  function initCanvas(): void {
    elCanvas.width = elImage.width;
    elCanvas.height = elImage.height;
    context.drawImage(elImage, 0, 0, elCanvas.width, elCanvas.height);
  }

  onMount(async () => {
    await init();
    context = elCanvas.getContext("2d");
    initCanvas();
  });

  async function predict(): Promise<void> {
    const buffer = tf.browser.fromPixels(elImage).div(255);
    const [width, height] = buffer.shape;
    const resize = tf.image.resizeBilinear(buffer, [width, height]);
    const cast = resize.cast("int32");
    const expand = cast.expandDims(0);
    const tensor = expand;
    const tensorImage = {
      tensor,
      inputShape: [buffer.shape[1], buffer.shape[0]],
      outputShape: tensor.shape,
      size: buffer.size,
    };
    const outputs = await model.executeAsync(tensorImage.tensor);
    // const arrays = !Array.isArray(outputs) ? outputs.array() : Promise.all(outputs.map(tensor => tensor.array()));
    // const predictions = await arrays;
    for (const prediction of outputs) {
      prediction.print();
    }
  }

  // function visualize_boxes_and_labels_on_image_array({
  //   image,
  //   boxes,
  //   classes,
  //   scores,
  //   instance_masks = null,
  //   instance_boundaries = null,
  //   keypoints = null,
  //   keypoint_scores = null,
  //   keypoint_edges = null,
  //   track_ids = null,
  //   use_normalized_coordinates = false,
  //   max_boxes_to_draw = 20,
  //   min_score_thresh = 0.5,
  //   agnostic_mode = false,
  //   line_thickness = 4,
  //   mask_alpha = 0.4,
  //   groundtruth_box_visualization_color = "black",
  //   skip_boxes = false,
  //   skip_scores = false,
  //   skip_labels = false,
  //   skip_track_ids = false,
  // }: {
  //   image: Tensor<Rank>,
  //   boxes: Tensor<Rank>,
  //   classes: Tensor<Rank>,
  //   scores: Tensor<Rank>,
  //   instance_masks: Tensor<Rank>,
  //   instance_boundaries: Tensor<Rank>,
  //   keypoints: Tensor<Rank>,
  //   keypoint_scores: Tensor<Rank>,
  //   keypoint_edges: Tensor<Rank>,
  //   track_ids: Tensor<Rank>,
  //   use_normalized_coordinates: boolean,
  //   max_boxes_to_draw: number,
  //   min_score_thresh: number,
  //   agnostic_mode: boolean,
  //   line_thickness: number,
  //   mask_alpha: number,
  //   groundtruth_box_visualization_color: string,
  //   skip_boxes: boolean,
  //   skip_scores: boolean,
  //   skip_labels: boolean,
  //   skip_track_ids: boolean,
  // }): void {
  //   const box_to_display_str_map = [];
  //   const box_to_color_map = "";
  //   const box_to_instance_masks_map = {};
  //   const box_to_instance_boundaries_map = {};
  //   const box_to_keypoints_map = [];
  //   const box_to_keypoint_scores_map = [];
  //   const box_to_track_ids_map = {};
  //
  //   if (!max_boxes_to_draw) {
  //     max_boxes_to_draw = boxes.shape[0];
  //   }
  //
  //   for (let i = 0; i < boxes.shape[0].length; i++) {
  //     if (max_boxes_to_draw === box_to_color_map.length) {
  //       break;
  //     }
  //
  //     if (!scores || scores[i] > min_score_thresh) {
  //       const box = boxes[i];
  //       if (instance_masks) {
  //         box_to_instance_masks_map[box] = instance_masks[i];
  //       }
  //       if (instance_boundaries) {
  //         box_to_instance_boundaries_map[box] = instance_boundaries[i];
  //       }
  //       if (keypoints) {
  //         box_to_keypoints_map[box] = keypoints[i];
  //       }
  //       if (scores) {
  //         box_to_keypoint_scores_map[box] = scores[i];
  //       } else {
  //         let display_str = "";
  //         if (!skip_scores) {
  //           if (!display_str) {
  //             display_str = `${scores[i] * 100}`;
  //           } else {
  //             display_str = `${display_str}: ${scores[i] * 100}`;
  //           }
  //         }
  //         box_to_display_str_map[box] += display_str;
  //         box_to_color_map[box] = "DarkOrange";
  //       }
  //     }
  //   }

  // Draw all boxes onto image.
  //   for (const [box, color] of box_to_color_map) {
  //     const [ymin, xmin, ymax, xmax] = box;
  //     if (instance_masks) {
  //       draw_mask_on_image_array({
  //         image,
  //         mask: box_to_instance_boundaries_map[box],
  //         color
  //       });
  //     }
  //     if (instance_boundaries) {
  //       draw_mask_on_image_array({
  //         image,
  //         mask: box_to_instance_boundaries_map[box],
  //         color: "red",
  //         alpha: 1
  //       });
  //     }
  //     draw_bounding_box_on_image_array({
  //       image,
  //       ymin,
  //       xmin,
  //       ymax,
  //       xmax,
  //       color,
  //       thickness: line_thickness,
  //       display_str_list: box_to_display_str_map[box],
  //       use_normalized_coordinates
  //     });
  //
  //     if (keypoints) {
  //       draw_keypoints_on_image_array({
  //         image,
  //         keypoints: box_to_keypoints_map[box],
  //         color,
  //         radius: line_thickness / 2,
  //         use_normalized_coordinates
  //       });
  //     }
  //   }
  // }

  // function draw_mask_on_image_array({ image, mask, color = "red", alpha = 0.4 }: { image: Tensor<Rank>; mask: Tensor<Rank>; color: string; alpha: number }) {
  //
  // }
</script>

<main>
  <div>
    <!--    <video autoplay bind:this={elVideo} muted on:click={predict}></video>-->
  </div>
  <div><img alt="" bind:this={elImage} src={srcImage}/></div>
  <div>
    <canvas bind:this={elCanvas} on:click={predict}/>
  </div>
</main>

<style>
  :root {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans",
    "Helvetica Neue", sans-serif;
  }

  img {
    display: none;
  }

  :global(body) {
    margin: 0;
    padding: 0;
  }
</style>
