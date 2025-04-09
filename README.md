Set 19
---

### 1. Create a `courses` array and render using `*ngFor`

**app.component.ts**
```ts
export class AppComponent {
  courses = ['Angular', 'React', 'Vue'];
}
```

**app.component.html**
```html
<ul>
  <li *ngFor="let course of courses">{{ course }}</li>
</ul>
```

---

### 2. Apply multiple CSS classes using `ngClass`

**styles.css**
```css
.highlight {
  background-color: yellow;
  padding: 5px;
}

.bold {
  font-weight: bolder;
}
```

**app.component.html**
```html
<p [ngClass]="{ highlight: isHighlighted, bold: isBold }">
  Angular ngClass Directive Example
</p>

<button (click)="toggleHighlight()">Toggle Highlight</button>
<button (click)="toggleBold()">Toggle Bold</button>
```
**app.component.ts**
```ts
import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  imports: [CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'demo';
  isHighlighted = false;
  isBold = false;

  toggleHighlight() {
    this.isHighlighted = !this.isHighlighted;
  }
  toggleBold() {
    this.isBold =!this.isBold;
  }
}

```
---

### 3. Display product code in lowercase and product name in uppercase

**app.component.ts**
```ts
export class AppComponent {
  productCode = 'XYZ123';
  productName = 'sample product';
}
```

**app.component.html**
```html
<p>Code: {{ productCode | lowercase }}</p>
<p>Name: {{ productName | uppercase }}</p>
```

---
