<script lang="ts">
  import { onMount } from "svelte";
  import "@tensorflow/tfjs-backend-cpu";
  import * as tf from "@tensorflow/tfjs-core";
  import * as tflite from "@tensorflow/tfjs-tflite";
  import type { TFLiteModel } from "@tensorflow/tfjs-tflite";

  let elVideo: HTMLVideoElement;
  let elCanvas: HTMLCanvasElement;
  let context: CanvasRenderingContext2D;
  let model: TFLiteModel;
  let stream: MediaStream;
  let elImage: HTMLImageElement;

  async function initWebCam(): Promise<void> {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment"
      }
    });
    elVideo.srcObject = stream;
  }

  async function init(): Promise<void> {
    model = await tflite.loadTFLiteModel("model.tflite");

    // await initWebCam();
  }

  async function initCanvas(): Promise<void> {
    elCanvas.width = elImage.width;
    elCanvas.height = elImage.height;
    context.drawImage(elImage, 0, 0, elCanvas.width, elCanvas.height);
  }

  onMount(async () => {
    await init();
    context = elCanvas.getContext("2d");
    await initCanvas();
  });

  async function predict(): Promise<void> {
    // context.drawImage(elVideo, 0, 0, elVideo.videoWidth, elVideo.videoHeight);
    const img = tf.browser.fromPixels(elImage);
    img.print(true);
    const input = tf.sub(tf.div(tf.expandDims(img), 127.5), 1);
    const output = model.predict(input) as tf.Tensor;
    console.log(output.dataSync());
  }

</script>

<main>
  <div>
    <!--    <video autoplay bind:this={elVideo} muted on:click={predict}></video>-->
  </div>
  <div><img alt="" bind:this={elImage} src="sample1.jpg" /></div>
  <div>
    <canvas bind:this={elCanvas} on:click={predict}></canvas>
  </div>
</main>

<style>
  :root {
    font-family: 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  }

  video {
    max-width: 100%;
    max-height: 100%;
  }

  img {
    display: none;
  }

  :global(body) {
    margin: 0;
    padding: 0;
  }
</style>
