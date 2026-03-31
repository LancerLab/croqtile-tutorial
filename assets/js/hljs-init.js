document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('pre code.language-choreo, pre code.language-co').forEach(function(el) {
    hljs.highlightElement(el);
  });
});
