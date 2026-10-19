// Every page's text from a PDF, run by macOS's own osascript (JavaScript for
// Automation) from material.rs — so a Reader needs no Python and no extra
// download to read a PDF. PDFKit gives the text layer; a page with none (a
// scan) is rendered at 2.5× and read by Vision, macOS's own text recognition.
// Prints {"pages": [text per page], "ocr": pages that needed recognition}.
ObjC.import('Foundation'); ObjC.import('Quartz'); ObjC.import('Vision'); ObjC.import('AppKit');
function run(argv) {
  const path = argv[0];
  const doc = $.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(path));
  if (doc.isNil()) return JSON.stringify({ error: 'not a PDF' });
  const n = doc.pageCount;
  const pages = [];
  let ocr = 0;
  for (let i = 0; i < n; i++) {
    const page = doc.pageAtIndex(i);
    let text = ObjC.unwrap(page.string) || '';
    if (text.replace(/\s+/g, '').length < 20) {
      const bounds = page.boundsForBox($.kPDFDisplayBoxMediaBox);
      const scale = 2.5;
      const size = $.NSMakeSize(bounds.size.width * scale, bounds.size.height * scale);
      const img = page.thumbnailOfSizeForBox(size, $.kPDFDisplayBoxMediaBox);
      const cg = img.CGImageForProposedRectContextHints($(), $(), $());
      const req = $.VNRecognizeTextRequest.alloc.init;
      req.recognitionLevel = $.VNRequestTextRecognitionLevelAccurate;
      req.usesLanguageCorrection = true;
      const handler = $.VNImageRequestHandler.alloc.initWithCGImageOptions(cg, $());
      const err = Ref();
      const ok = handler.performRequestsError($.NSArray.arrayWithObject(req), err);
      if (ok) {
        const res = req.results;
        const lines = [];
        for (let j = 0; j < res.count; j++) {
          const cands = res.objectAtIndex(j).topCandidates(1);
          if (cands.count) lines.push(ObjC.unwrap(cands.objectAtIndex(0).string));
        }
        text = lines.join('\n');
        ocr++;
      }
    }
    pages.push(text);
  }
  return JSON.stringify({ pages, ocr });
}
