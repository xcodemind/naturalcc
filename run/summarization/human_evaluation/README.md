## jquery-starrating

An extremely lightweight star-rating jQuery plugin

[Demo](https://codepen.io/zanderwar/pen/XZbjdX)

## Installation
```bash
bower install --save starrating.js
```

## Usage
HTML:
```html
<ul id="starRating"></ul>
```

Javascript:
```javascript
(function( $ ) {
    $('#starRating').starRating(/* options */)
}(jQuery))
```

## Options
| Option  | Description                                                    |
|---------|----------------------------------------------------------------|
| stars   | Set the amount of stars that will be generated: **Default: 5** |
| current | Set the current amount of stars that should be active          |

### Example
```javascript
$('#starRating').starRating({
    stars: 10,
    current: 3
}) 
```

Alternatively you can add `data-stars="10"` and/or `data-current="3"` to your element to achieve the same