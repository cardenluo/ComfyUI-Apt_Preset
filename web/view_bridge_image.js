import { app } from '../../../scripts/app.js'
// import { ComfyDialog } from '../../../scripts/ui.js'
import { api } from '../../../scripts/api.js'

const MASK_EDITOR_IMAGE = /^clipspace-painted-masked-\d+\.png$/i

function isMaskEditorImage (image) {
  return image?.filename && MASK_EDITOR_IMAGE.test(image.filename) && (image.type || 'input') === 'input'
}

function imageUrl (image) {
  if (!image?.filename) return ''
  const params = new URLSearchParams({
    filename: image.filename,
    type: image.type || 'temp',
    subfolder: image.subfolder || '',
    rand: String(Date.now())
  })
  return api.apiURL(`/view?${params.toString()}`)
}

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
            const editedImages = (node.images || []).filter(isMaskEditorImage)
            if (editedImages.length > 0) node.__viewBridgeEditedImages = editedImages
            if (node.__viewBridgeEditedImages?.length > 0) return { images: node.__viewBridgeEditedImages }
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
      
      // 隐藏 image_update widget
      if (nodeData.name === 'view_bridge_image') {
        requestAnimationFrame(() => {
          const imageUpdateWidget = this.widgets?.find(w => w.name === 'image_update')
          if (imageUpdateWidget) {
            // 通过设置computeSize来隐藏widget
            imageUpdateWidget.computeSize = () => [0, -4]
            imageUpdateWidget.disabled = true
          }
        })
      }
    }

    if (nodeData.name === 'view_bridge_image') {
      const origOnExecuted = nodeType.prototype.onExecuted
      nodeType.prototype.onExecuted = function (message) {
        const result = origOnExecuted?.apply(this, arguments)
        const sourceImage = message?.source_images?.[0]
        if (sourceImage) this.__viewBridgeSourcePreviewUrl = imageUrl(sourceImage)
        if (message?.clear_image_update?.at?.(-1) === true) {
          this.__viewBridgeEditedImages = null
          const imageUpdateWidget = this.widgets?.find(w => w.name === 'image_update')
          if (imageUpdateWidget) imageUpdateWidget.value = {}
        }
        window.dispatchEvent(new CustomEvent('apt-media-source-preview-change', { detail: { nodeId: Number(this.id) } }))
        return result
      }
    }
  },
  async loadedGraphNode (node, app) {
    if (node.type === 'view_bridge_image') {
      const image_update = node.widgets.filter(w => w.name === 'image_update')[0]

      //从ComfyUI\web\scripts\widgets.js，IMAGEUPLOAD参考代码
      function showImage (file) {
        const { filename, subfolder = '', type = 'input' } = file
        const img = new Image()
        img.onload = () => {
          node.imgs = [img]
          app.graph.setDirtyCanvas(true)
        }
        img.src = api.apiURL(
          `/view?filename=${encodeURIComponent(
            filename
          )}&type=${encodeURIComponent(type)}&subfolder=${encodeURIComponent(subfolder)}${app.getPreviewFormatParam()}${app.getRandParam()}`
        )
        node.setSizeForImage?.()
      }

      const editedImages = (image_update?.value?.images || []).filter(isMaskEditorImage)
      if (editedImages.length > 0) {
        node.__viewBridgeEditedImages = editedImages
        showImage(editedImages[0])
        node.images = editedImages
      } else if (image_update) {
        image_update.value = {}
      }
    }
  }
})
//#endregion
