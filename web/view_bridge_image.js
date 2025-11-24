import { app } from '../../../scripts/app.js'
// import { ComfyDialog } from '../../../scripts/ui.js'
import { api } from '../../../scripts/api.js'


//#region 
app.registerExtension({
  name: 'web_node.edit_mask.view_bridge_image',    
  async getCustomWidgets (app) {
    return {
      IMAGE_FILE (node, inputName, inputData, app) {
        const widget = {
          type: inputData[0], // the type, CHEESE
          name: inputName, // the name, slice
          size: [128, 88], // a default size
          draw (ctx, node, width, y) {},
          callback: e => console.log(e),
          computeSize (...args) {
            return [128, 24] // a method to compute the current size of the widget
          },
          async serializeValue (nodeId, widgetIndex) {
            // node=app.graph.getNodeById(node.id)
            if (node.images) return { images: node.images }
            return {}
          }
        }
        node.addCustomWidget(widget)
        return widget
      }
    }
  },

  async beforeRegisterNodeDef (nodeType, nodeData, app) {
    // 节点创建之后的事件回调
    const orig_nodeCreated = nodeType.prototype.onNodeCreated
    nodeType.prototype.onNodeCreated = function () {
      orig_nodeCreated?.apply(this, arguments)
    }
  },
  async loadedGraphNode (node, app) {
    if (node.type === 'view_bridge_image') {
      let image_update = node.widgets.filter(w => w.name === 'image_update')[0]
      console.log('#image_update', image_update)

      //从ComfyUI\web\scripts\widgets.js，IMAGEUPLOAD参考代码
      function showImage (file) {
        const { filename, subfolder } = file
        const img = new Image()
        img.onload = () => {
          node.imgs = [img]
          app.graph.setDirtyCanvas(true)
        }
        img.src = api.apiURL(
          `/view?filename=${encodeURIComponent(
            filename
          )}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`
        )
        node.setSizeForImage?.()
      }

      if (image_update.value?.images?.length > 0)
        showImage(image_update.value.images[0])
        // 数据要写入到节点
        node.images=image_update.value.images;
    }
  }
})
//#endregion