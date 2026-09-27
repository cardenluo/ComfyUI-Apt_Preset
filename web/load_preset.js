import { ComfyApp, app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

var allow_set_flag = true;

app.registerExtension({
  name: "Zpreset",

  async beforeConfigureGraph() {
    allow_set_flag = false;
  },

  async nodeCreated(node) {
    function send_message(node_id, message) {
      console.log("sendMessage");
      const body = new FormData();
      body.append("message", message);
      body.append("node_id", node_id);
      api.fetchApi("/Apt_Preset_path", { method: "POST", body });
    }

    //#region

    if (node.comfyClass == "sum_load_adv") {
      Object.defineProperty(node.widgets[0], "value", {
        set: (value) => {
          node._value = value;
          console.log("set");

          if (allow_set_flag) send_message(node.id, value);
        },
        get: () => {
          return node._value;
        },
      });

      function messageHandler(event) {
        if (node.id == event.detail.node) {
          node.widgets[1].value = event.detail.message["ckpt_name"]
            ? event.detail.message["ckpt_name"]
            : "AA\\majicMIX realistic _v1.safetensors";
          node.widgets[2].value = event.detail.message["unet_name"]
            ? event.detail.message["unet_name"]
            : "None";
          node.widgets[3].value = event.detail.message["unet_Weight_Dtype"]
            ? event.detail.message["unet_Weight_Dtype"]
            : "None";
          node.widgets[4].value = event.detail.message["clip_type"]
            ? event.detail.message["clip_type"]
            : "None";
          node.widgets[5].value = event.detail.message["clip1"]
            ? event.detail.message["clip1"]
            : "None";
          node.widgets[6].value = event.detail.message["clip2"]
            ? event.detail.message["clip2"]
            : "None";
          node.widgets[7].value = event.detail.message["guidance"]
            ? event.detail.message["guidance"]
            : "3.5";
          node.widgets[8].value = event.detail.message["clip3"]
            ? event.detail.message["clip3"]
            : "None";
          node.widgets[9].value = event.detail.message["clip4"]
            ? event.detail.message["clip4"]
            : "None";
          node.widgets[10].value = event.detail.message["vae"]
            ? event.detail.message["vae"]
            : "A-840000-ema-pruned-real.pt";
          node.widgets[11].value = event.detail.message["lora"]
            ? event.detail.message["lora"]
            : "None";
          node.widgets[12].value = event.detail.message["lora_strength"]
            ? event.detail.message["lora_strength"]
            : "1.0";
          node.widgets[13].value = event.detail.message["width"]
            ? event.detail.message["width"]
            : "512";
          node.widgets[14].value = event.detail.message["height"]
            ? event.detail.message["height"]
            : "512";
          node.widgets[15].value = event.detail.message["steps"]
            ? event.detail.message["steps"]
            : "20";
          node.widgets[16].value = event.detail.message["cfg"]
            ? event.detail.message["cfg"]
            : "8";
          node.widgets[17].value = event.detail.message["sampler"]
            ? event.detail.message["sampler"]
            : "euler a";
          node.widgets[18].value = event.detail.message["scheduler"]
            ? event.detail.message["scheduler"]
            : "normal";
          node.widgets[19].value = event.detail.message["positive"]
            ? event.detail.message["positive"]
            : "beautiful detailed glow,a girl";
          node.widgets[20].value = event.detail.message["negative"]
            ? event.detail.message["negative"]
            : "worst quality, low quality";
        }
      }
      api.addEventListener("my.custom.message", messageHandler);
    }

    if (node.comfyClass == "load_basic") {
      Object.defineProperty(node.widgets[0], "value", {
        set: (value) => {
          node._value = value;
          console.log("set");

          if (allow_set_flag) send_message(node.id, value);
        },
        get: () => {
          return node._value;
        },
      });

      function messageHandler(event) {
        if (node.id == event.detail.node) {
          node.widgets[1].value = event.detail.message["ckpt_name"]
            ? event.detail.message["ckpt_name"]
            : "AA\\majicMIX realistic _v1.safetensors";
          node.widgets[2].value = event.detail.message["clipnum"]
            ? event.detail.message["clipnum"]
            : "-2";
          node.widgets[3].value = event.detail.message["vae"]
            ? event.detail.message["vae"]
            : "A-840000-ema-pruned-real.pt";
          node.widgets[4].value = event.detail.message["lora"]
            ? event.detail.message["lora"]
            : "None";
          node.widgets[5].value = event.detail.message["lora_strength"]
            ? event.detail.message["lora_strength"]
            : "1.0";
          node.widgets[6].value = event.detail.message["width"]
            ? event.detail.message["width"]
            : "512";
          node.widgets[7].value = event.detail.message["height"]
            ? event.detail.message["height"]
            : "512";
          node.widgets[8].value = event.detail.message["batch"]
            ? event.detail.message["batch"]
            : "1";
          node.widgets[9].value = event.detail.message["steps"]
            ? event.detail.message["steps"]
            : "20";
          node.widgets[10].value = event.detail.message["cfg"]
            ? event.detail.message["cfg"]
            : "8";
          node.widgets[11].value = event.detail.message["sampler"]
            ? event.detail.message["sampler"]
            : "euler a";
          node.widgets[12].value = event.detail.message["scheduler"]
            ? event.detail.message["scheduler"]
            : "normal";
          node.widgets[13].value = event.detail.message["positive"]
            ? event.detail.message["positive"]
            : "beautiful detailed glow,a girl";
          node.widgets[14].value = event.detail.message["negative"]
            ? event.detail.message["negative"]
            : "worst quality, low quality";
        }
      }
      api.addEventListener("my.custom.message", messageHandler);
    }

    if (node.comfyClass == "load_FLUX") {
      Object.defineProperty(node.widgets[0], "value", {
        set: (value) => {
          node._value = value;
          console.log("set");

          if (allow_set_flag) send_message(node.id, value);
        },
        get: () => {
          return node._value;
        },
      });

      function messageHandler(event) {
        if (node.id == event.detail.node) {
          node.widgets[1].value = event.detail.message["unet_name"]
            ? event.detail.message["unet_name"]
            : "None";
          node.widgets[2].value = event.detail.message["unet_Weight_Dtype"]
            ? event.detail.message["unet_Weight_Dtype"]
            : "None";
          node.widgets[3].value = event.detail.message["clip_type"]
            ? event.detail.message["clip_type"]
            : "None";
          node.widgets[4].value = event.detail.message["clip1"]
            ? event.detail.message["clip1"]
            : "None";
          node.widgets[5].value = event.detail.message["clip2"]
            ? event.detail.message["clip2"]
            : "None";
          node.widgets[6].value = event.detail.message["guidance"]
            ? event.detail.message["guidance"]
            : "3.5";
          node.widgets[7].value = event.detail.message["vae"]
            ? event.detail.message["vae"]
            : "None";
          node.widgets[8].value = event.detail.message["lora"]
            ? event.detail.message["lora"]
            : "None";
          node.widgets[9].value = event.detail.message["lora_strength"]
            ? event.detail.message["lora_strength"]
            : "1.0";
          node.widgets[10].value = event.detail.message["width"]
            ? event.detail.message["width"]
            : "1024";
          node.widgets[11].value = event.detail.message["height"]
            ? event.detail.message["height"]
            : "1024";
          node.widgets[12].value = event.detail.message["batch"]
            ? event.detail.message["batch"]
            : "1";
          node.widgets[13].value = event.detail.message["steps"]
            ? event.detail.message["steps"]
            : "20";
          node.widgets[14].value = event.detail.message["cfg"]
            ? event.detail.message["cfg"]
            : "8";
          node.widgets[15].value = event.detail.message["sampler"]
            ? event.detail.message["sampler"]
            : "euler a";
          node.widgets[16].value = event.detail.message["scheduler"]
            ? event.detail.message["scheduler"]
            : "normal";
          node.widgets[17].value = event.detail.message["positive"]
            ? event.detail.message["positive"]
            : "beautiful detailed glow,a girl";
        }
      }
      api.addEventListener("my.custom.message", messageHandler);
    }


    if (node.comfyClass == "load_SD35") {
      Object.defineProperty(node.widgets[0], "value", {
        set: (value) => {
          node._value = value;
          console.log("set");

          if (allow_set_flag) send_message(node.id, value);
        },
        get: () => {
          return node._value;
        },
      });

      function messageHandler(event) {
        if (node.id == event.detail.node) {
          node.widgets[1].value = event.detail.message["unet_name"]
            ? event.detail.message["unet_name"]
            : "None";
          node.widgets[2].value = event.detail.message["unet_Weight_Dtype"]
            ? event.detail.message["unet_Weight_Dtype"]
            : "None";
          node.widgets[3].value = event.detail.message["clip1"]
            ? event.detail.message["clip1"]
            : "None";
          node.widgets[4].value = event.detail.message["clip2"]
            ? event.detail.message["clip2"]
            : "None";
          node.widgets[5].value = event.detail.message["clip3"]
            ? event.detail.message["clip3"]
            : "None";
          node.widgets[6].value = event.detail.message["vae"]
            ? event.detail.message["vae"]
            : "None";
          node.widgets[7].value = event.detail.message["lora"]
            ? event.detail.message["lora"]
            : "None";
          node.widgets[8].value = event.detail.message["lora_strength"]
            ? event.detail.message["lora_strength"]
            : "1.0";
          node.widgets[9].value = event.detail.message["width"]
            ? event.detail.message["width"]
            : "1024";
          node.widgets[10].value = event.detail.message["height"]
            ? event.detail.message["height"]
            : "1024";
          node.widgets[11].value = event.detail.message["batch"]
            ? event.detail.message["batch"]
            : "1";
          node.widgets[12].value = event.detail.message["steps"]
            ? event.detail.message["steps"]
            : "20";
          node.widgets[13].value = event.detail.message["cfg"]
            ? event.detail.message["cfg"]
            : "8";
          node.widgets[14].value = event.detail.message["sampler"]
            ? event.detail.message["sampler"]
            : "euler a";
          node.widgets[15].value = event.detail.message["scheduler"]
            ? event.detail.message["scheduler"]
            : "normal";

          node.widgets[16].value = event.detail.message["positive"]
            ? event.detail.message["positive"]
            : "beautiful detailed glow,a girl";
        }
      }
      api.addEventListener("my.custom.message", messageHandler);
    }


    if (node.comfyClass == "load_Nanchaku") {
      Object.defineProperty(node.widgets[0], "value", {
        set: (value) => {
          node._value = value;
          console.log("set");

          if (allow_set_flag) send_message(node.id, value);
        },
        get: () => {
          return node._value;
        },
      });

    function messageHandler(event) {
      if (node.id == event.detail.node) {
        node.widgets[1].value = event.detail.message["unet_name"] ? event.detail.message["unet_name"] : "None";     
        node.widgets[2].value = event.detail.message["cache_threshold"] ? event.detail.message["cache_threshold"] : "0.00";
        node.widgets[3].value = event.detail.message["attention"] ? event.detail.message["attention"] : "None";
        node.widgets[4].value = event.detail.message["cpu_offload"] ? event.detail.message["cpu_offload"] : "None";        
        node.widgets[5].value = event.detail.message["clip1"] ? event.detail.message["clip1"] : "None";
        node.widgets[6].value = event.detail.message["clip2"] ? event.detail.message["clip2"] : "None";
        node.widgets[7].value = event.detail.message["guidance"] ? event.detail.message["guidance"] : "3.5";
        node.widgets[8].value = event.detail.message["vae"] ? event.detail.message["vae"] : "None";
        node.widgets[9].value = event.detail.message["lora"]? event.detail.message["lora"]: "None";
        node.widgets[10].value = event.detail.message["lora_strength"]? event.detail.message["lora_strength"]: "1.0";
        node.widgets[11].value = event.detail.message["width"] ? event.detail.message["width"] : "1024";
        node.widgets[12].value = event.detail.message["height"] ? event.detail.message["height"] : "1024";
        node.widgets[13].value = event.detail.message["steps"] ? event.detail.message["steps"] : "8";
        node.widgets[14].value = event.detail.message["cfg"] ? event.detail.message["cfg"] : "1";
        node.widgets[15].value = event.detail.message["sampler"] ? event.detail.message["sampler"] : "None";
        node.widgets[16].value = event.detail.message["scheduler"] ? event.detail.message["scheduler"] : "None";
      }

    }

      api.addEventListener("my.custom.message", messageHandler);
    }

    //#endregion
  },

  async afterConfigureGraph() {
    allow_set_flag = true;
  },
});






































