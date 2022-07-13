<script lang="ts">
  import type { GraphModel, io } from "@tensorflow/tfjs";
  import * as tf from "@tensorflow/tfjs";
  import { browser } from "@tensorflow/tfjs";
  import { onMount } from "svelte";

  async function init(): Promise<void> {
    model = await tf.loadGraphModel("./web_model/model.json");

    elVideo.srcObject = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
      }
    });
  }

  onMount(async () => {
    await init();
    context = elVideo.getContext("2d");
  });

  async function predict(): Promise<void> {
    context.drawImage(elVideo, 0, 0, 500, 500);

    const tensor = browser.fromPixels(elCanvas)
      .resizeBilinear([500, 500])
      .toFloat()
      .expandDims();
    const predictions = (await model.predict(tensor)).data();
    console.log({ predictions });
  }

  let elVideo;
  let elCanvas: HTMLCanvasElement;
  let context: CanvasRenderingContext2D;
  let model: GraphModel<string | io.IOHandler>;
</script>

<main>
  <video bind:this={elVideo} muted autoplay on:click={predict}></video>
  <canvas bind:this={elCanvas}></canvas>
</main>

<style>
  :root {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen,
    Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
  }

  main {
    text-align: center;
    padding: 1em;
    margin: 0 auto;
  }

</style>
